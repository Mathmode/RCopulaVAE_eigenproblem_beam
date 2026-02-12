import tensorflow as tf
import sys
import os
import ctypes

def find_and_load_zlib():
    """
    Hunts for zlibwapi.dll in specific locations and tries to load it 
    by full path to verify compatibility.
    """
    print("Diagnostics: Hunting for zlibwapi.dll...")
    
    # 1. List of places to look
    search_paths = [
        # The standard CUDA paths
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin",
        # The user's specific Miniconda path mentioned earlier
        r"C:\Users\anafd\miniconda3\pkgs\cudatoolkit-11.2.2-h7d7167e_13\Library\bin",
        # Current directory
        os.getcwd(),
    ]
    
    target_dll = "zlibwapi.dll"
    found_path = None
    
    # 2. Search for the file existence first
    for path in search_paths:
        full_path = os.path.join(path, target_dll)
        if os.path.exists(full_path):
            print(f"   🔎 Found file at: {full_path}")
            try:
                # 3. Try to load it explicitly
                ctypes.WinDLL(full_path)
                print(f"   ✅ successfully loaded via absolute path!")
                found_path = path
                break
            except OSError as e:
                print(f"   ❌ File exists but failed to load (Corrupt/Wrong Arch?): {e}")

    # 4. If found, configure the environment for the rest of the script
    if found_path:
        print("\n   Applying fix for this session...")
        os.environ['PATH'] = found_path + os.pathsep + os.environ['PATH']
        try:
            os.add_dll_directory(found_path)
            print("   -> Added via os.add_dll_directory()")
        except:
            pass
        return found_path
    else:
        return None

def check_tf_gpu():
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"TensorFlow Version: {tf.__version__}")
    print("-" * 30)
    
    # --- STEP 1: RESOLVE DLLS ---
    if sys.platform == 'win32':
        valid_dll_path = find_and_load_zlib()
        
        # Check if loadable now
        try:
            ctypes.WinDLL('zlibwapi.dll')
            print("   ✅ Global DLL check passed.")
        except OSError:
            print("\n" + "="*60)
            print("⚠️  CRITICAL FAILURE: zlibwapi.dll could not be loaded.")
            print("="*60)
            print("   The script looked in common folders but failed.")
            print("   \n   LAST RESORT SOLUTION (100% Works):")
            print("   1. Go to your Miniconda folder where you found the file:")
            print(r"      C:\Users\anafd\miniconda3\pkgs\cudatoolkit-11.2.2-h7d7167e_13\Library\bin")
            print("   2. Copy 'zlibwapi.dll'")
            print("   3. Paste it into: C:\Windows\System32")
            print("   4. Restart Python.")
            return

    # --- STEP 2: TENSORFLOW TESTS ---
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print("\nStarting TensorFlow checks...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Detected: {len(gpus)}")
        except RuntimeError:
            pass

        print("Running Linear Algebra Test (cuSolver)... ", end="")
        try:
            with tf.device('/GPU:0'):
                matrix = tf.constant([[4.0, 1.0], [1.0, 4.0]])
                L = tf.linalg.cholesky(matrix) 
                print("PASSED ✅")
                
                print("\n" + "*" * 50)
                print("🎉 IT WORKS! COPY THE CODE BELOW TO YOUR MAIN SCRIPT:")
                print("*" * 50)
                print("import os")
                if valid_dll_path:
                    # Escape backslashes for the print output
                    safe_path = valid_dll_path.replace('\\', '\\\\')
                    print(f"os.add_dll_directory(r'{valid_dll_path}')")
                else:
                    print("# (No specific path found, ensure zlibwapi.dll is in System32)")
                print("import tensorflow as tf")
                print("# ... rest of your code ...")
                print("*" * 50)
                
        except RuntimeError as e:
            print(f"FAILED ❌\n   Error: {e}")
    else:
        print("❌ No GPU detected.")

if __name__ == "__main__":
    check_tf_gpu()