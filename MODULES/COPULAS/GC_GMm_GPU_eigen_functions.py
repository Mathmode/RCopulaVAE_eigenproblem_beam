import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

# -------------------------------------------------------------------------
# 1. VECTORIZED ASSEMBLY (Optimized & JIT Compiled)
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def assemble_global_Kmatrices(Ke_matrices_dam, n_elements, num_samples, fixed_dofs_indices=None):
    """
    Vectorized assembly of global stiffness matrices.
    """
    # 1. Cast and Shapes
    Ke_matrices_dam = tf.cast(Ke_matrices_dam, dtype=tf.float32)
    batch_size_total = tf.shape(Ke_matrices_dam)[0] 
    n_nodes = n_elements + 1
    n_dofs = 2 * n_nodes
    
    # 2. Flatten Element Matrices -> (Batch, Elem*16)
    K_values_flat = tf.reshape(Ke_matrices_dam, [batch_size_total, -1]) 

    # 3. Precompute Indices (ONCE, assumed constant topology)
    indices_list = []
    for i in range(n_elements):
        # Standard Beam Element Formulation: [v1, th1, v2, th2]
        node_i = i
        node_j = i + 1
        global_dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1] 
        
        rows = np.repeat(global_dofs, 4)
        cols = np.tile(global_dofs, 4)
        elem_indices = np.stack([rows, cols], axis=1)
        indices_list.append(elem_indices)
        
    base_indices = np.concatenate(indices_list, axis=0)
    base_indices_tf = tf.constant(base_indices, dtype=tf.int32)
    
    # 4. Vectorized Scatter Indices
    num_vals = n_elements * 16
    batch_indices = tf.repeat(tf.range(batch_size_total), num_vals)
    tiled_base_indices = tf.tile(base_indices_tf, [batch_size_total, 1])
    
    full_indices = tf.concat([
        tf.expand_dims(batch_indices, 1), 
        tf.cast(tiled_base_indices, tf.int32)
    ], axis=1)
    
    flat_values = tf.reshape(K_values_flat, [-1])
    
    # Scatter into global matrix
    K_global = tf.scatter_nd(full_indices, flat_values, shape=[batch_size_total, n_dofs, n_dofs])
    
    # 5. Extract Free DOFs
    if fixed_dofs_indices is None:
        fixed_dofs_tf = tf.stack([0, n_dofs - 2])
        fixed_dofs_tf = tf.cast(fixed_dofs_tf, dtype=tf.int32)
    else:
        fixed_dofs_tf = tf.cast(fixed_dofs_indices, dtype=tf.int32)

    # Boolean Mask for Free DOFs
    all_indices = tf.range(n_dofs, dtype=tf.int32)
    is_fixed = tf.equal(tf.expand_dims(all_indices, 1), tf.expand_dims(fixed_dofs_tf, 0))
    is_fixed_mask = tf.reduce_any(is_fixed, axis=1)
    free_dofs_tf = tf.boolean_mask(all_indices, ~is_fixed_mask)
    
    # Slice K_global to get K_free
    K_free = tf.gather(K_global, free_dofs_tf, axis=1)
    K_free = tf.gather(K_free, free_dofs_tf, axis=2)
    
    return K_free

# -------------------------------------------------------------------------
# 2. ROBUST EIGENSOLVER LAYER (With Sign Consistency)
# -------------------------------------------------------------------------

@tf.function(jit_compile=True)
def safe_eigh(tensor):
    return tf.linalg.eigh(tensor)

class Solve_eigenproblem(Layer):
    def __init__(self, num_dofs_total, n_modes, L_inv, fixed_dofs_indices, **kwargs):
        """
        Args:
            num_dofs_total: Total DOFs (2 * Nodes)
            n_modes: Number of modes to output
            L_inv: Inverse Cholesky of Mass Matrix (Free DOFs)
            fixed_dofs_indices: Indices of supports
        """
        super(Solve_eigenproblem, self).__init__(**kwargs)
        self.num_dofs_total = num_dofs_total
        self.n_modes = n_modes
        
        # Calculate Free DOFs for reconstruction
        if tf.is_tensor(fixed_dofs_indices):
            fixed_dofs_np = fixed_dofs_indices.numpy()
        else:
            fixed_dofs_np = np.array(fixed_dofs_indices) if fixed_dofs_indices is not None else np.array([0, num_dofs_total-2])
            
        all_indices = np.arange(num_dofs_total)
        self.free_dofs = np.delete(all_indices, fixed_dofs_np)
        self.fixed_dofs = fixed_dofs_np
        
        self.L_inv = tf.Variable(
            initial_value=tf.cast(L_inv, dtype=tf.float32),
            trainable=True,
            name="L_inverse_matrix"
        )
        
    def call(self, inputs):
        """
        Input: Kfree_matrices (Batch, N_free, N_free)
        Output: freqs_hz, rot_modes, vert_modes (Transposed to [Batch, Modes, Nodes])
        """
        Kfree = tf.cast(inputs, dtype=tf.float32)
        batch_size = tf.shape(Kfree)[0]
        
        # --- Solve Generalized Eigenproblem ---
        # A = L^-T * K * L^-1
        L_inv_T = tf.transpose(self.L_inv)
        half_transformed = tf.matmul(Kfree, self.L_inv)
        A = tf.matmul(L_inv_T, half_transformed)

        # Jitter for stability
        jitter = 1e-6 * tf.eye(tf.shape(A)[1], batch_shape=[batch_size])
        A = A + jitter
        
        eigvals, eigvecs_transformed = safe_eigh(A)
        
        # --- Filter & Frequencies ---
        eigvals = eigvals[:, :self.n_modes]
        eigvecs_transformed = eigvecs_transformed[:, :, :self.n_modes]
        
        eigvals = tf.nn.relu(eigvals) 
        freqs_rad = tf.sqrt(eigvals + 1e-12)
        freqs_hz = freqs_rad / (2.0 * np.pi)
        
        # --- Recover Physical Modes (Free DOFs) ---
        physical_modes_free = tf.matmul(self.L_inv, eigvecs_transformed)

        # --- Reconstruct Full System ---
        flat_modes = tf.reshape(physical_modes_free, [-1, self.n_modes])
        batch_indices = tf.repeat(tf.range(batch_size), len(self.free_dofs))
        dof_indices = tf.tile(tf.constant(self.free_dofs, dtype=tf.int32), [batch_size])
        scatter_indices = tf.stack([batch_indices, dof_indices], axis=1) 
        
        physical_modes_full = tf.scatter_nd(
            indices=scatter_indices,
            updates=flat_modes,
            shape=[batch_size, self.num_dofs_total, self.n_modes]
        )
        
        # --- Separate Components ---
        vert_modes = physical_modes_full[:, 0::2, :]
        rot_modes = physical_modes_full[:, 1::2, :]
        
        # --- Sign Consistency ---
        # 1. Find index of max absolute value (returns int64)
        max_indices = tf.argmax(tf.abs(vert_modes), axis=1)
        
        # 2. THE FIX: Explicitly cast int64 -> int32 so it matches tf.range()
        max_indices = tf.cast(max_indices, dtype=tf.int32) 
        
        batch_idx_grid = tf.broadcast_to(tf.expand_dims(tf.range(batch_size), 1), tf.shape(max_indices))
        mode_idx_grid = tf.broadcast_to(tf.expand_dims(tf.range(self.n_modes), 0), tf.shape(max_indices))
        
        # 3. Stack now receives all int32 tensors
        gather_indices = tf.stack([batch_idx_grid, max_indices, mode_idx_grid], axis=2)
        
        peak_vals = tf.gather_nd(vert_modes, gather_indices) 
        signs = tf.sign(peak_vals)
        signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
        signs_expanded = tf.expand_dims(signs, axis=1)
        
        vert_modes = vert_modes * signs_expanded
        rot_modes = rot_modes * signs_expanded

        # --- Transpose to [Batch, Modes, Nodes] ---
        vert_modes = tf.transpose(vert_modes, perm=[0, 2, 1])
        rot_modes = tf.transpose(rot_modes, perm=[0, 2, 1])
        
        return freqs_hz, rot_modes, vert_modes

