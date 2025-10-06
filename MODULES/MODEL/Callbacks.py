from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import Callback

class Loss_Dependent_Adaptive_Learning_Rate(Callback):

    def __init__(self,
                 printing_frequency = 0,
                 lr_down_factor = 0.9,
                 loss_down_threshold_factor = 0.95,
                 lr_up_factor = 1.01,
                 loss_memory = 100,
                 best_loss_improvement_proportion = 10**(-3)):

        super(Loss_Dependent_Adaptive_Learning_Rate, self).__init__()

        self.printing_frequency = printing_frequency
        self.lr_down_factor = lr_down_factor
        self.loss_down_threshold_factor = loss_down_threshold_factor
        self.lr_up_factor = lr_up_factor
        self.loss_memory = loss_memory
        self.best_loss_improvement_proportion = best_loss_improvement_proportion

        self.best_loss_history = list()
        self.best_loss_improvements_history = list()

    def on_epoch_begin(self, epoch, logs={}):

        if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0:
            print("\nEPOCH", epoch + 1)
        # In the first iteration
        if not epoch > 0:
            self.w_epoch = list()
            for layer in self.model.layers:
                self.w_epoch.append(layer.get_weights())

            self.w_best = self.w_epoch
            self.bool_previously_weights_accepted = True

    def on_epoch_end(self, epoch, logs={}):

        self.loss_epoch = logs.get('loss')
        self.lr_epoch = K.get_value(self.model.optimizer.lr)

        self.w_epoch_plus_one = list()
        for layer in self.model.layers:
            self.w_epoch_plus_one.append(layer.get_weights())

        if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0 and epoch > 0:
            print("\tBEFORE")
            print("\tLoss Best:", self.loss_best)
            print("\tLoss {}:".format(epoch+1), self.loss_epoch)
            print("\tLr   {}:".format(epoch+1), self.lr_epoch)

        # In the first iteration
        if not epoch > 0:
            self.loss_best = self.loss_epoch

            if self.printing_frequency:
                print("\nEPOCH 1")
                print("\tLoss Best: ", self.loss_best)
                print("\tLoss 0: ", self.loss_epoch)
                print("\tLr   1: ", self.lr_epoch)

        else:
            # If the loss of the previous model is worse
            if self.loss_epoch > 0.9999 * self.loss_best and self.bool_previously_weights_accepted:

                # We decrease the learning rate
                K.set_value(self.model.optimizer.lr, self.lr_epoch * self.lr_down_factor)

                # We reject the weights
                self.w_epoch_plus_one = self.w_best
                for i in range(len(self.w_epoch_plus_one)):
                    self.model.layers[i].set_weights(self.w_epoch_plus_one[i])

                if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0 and epoch > 0:
                    print("\n\t*REJECTED WEIGHTS*")
                    print("\t*Decrease Learning rate*\n")

                self.bool_previously_weights_accepted = False
                self.loss_epoch = self.loss_best

            else:
                # We accept the weights
                if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0 and epoch > 0:
                    print("\n\t*ACCEPTED WEIGHTS*")

                self.w_best = self.w_epoch

                # If the convergence is slow
                if abs(self.loss_epoch) > self.loss_down_threshold_factor * abs(self.loss_best) and self.bool_previously_weights_accepted:
                    # We increase the learning rate
                    K.set_value(self.model.optimizer.lr, self.lr_epoch * self.lr_up_factor)

                    if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0 and epoch > 0:
                        print("\t*Increase Learning Rate*\n")

                else:
                    self.bool_previously_weights_accepted = True
                    if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0 and epoch > 0:
                        print("\t*Maintain Learning Rate*\n")

                self.loss_best = self.loss_epoch

        if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0 and epoch > 0:
            print("\tAFTER")
            print("\tLoss Best:", self.loss_best)
            print("\tLoss {}:".format(epoch + 1), self.loss_epoch)
            print("\tLr   {}:".format(epoch + 2), K.get_value(self.model.optimizer.lr))

        self.w_epoch = self.w_epoch_plus_one

        self.best_loss_history.append(self.loss_best)

        ####### Early stopping depending on best losses #######

        if len(self.best_loss_history) >= self.loss_memory:
            improvement_proportion = 1-self.loss_best/self.best_loss_history[-self.loss_memory]
            self.best_loss_improvements_history.append(improvement_proportion)

            if self.printing_frequency and (epoch + 1) % self.printing_frequency == 0:
                print("\n\tLoss Best improvement:", improvement_proportion)

            if improvement_proportion < self.best_loss_improvement_proportion:
                self.model.stop_training = True
                print("\nSTOP TRAINING - EPOCH {}: loss improvement below a {} relative proportion after {} iterations".format(epoch+1, self.best_loss_improvement_proportion, self.loss_memory))

    def on_train_end(self, logs={}):

        # We update the model with the best weights whose loss is known
        for i in range(len(self.w_best)):
            self.model.layers[i].set_weights(self.w_best[i])

class Early_Stopping(Callback):
    def __init__(self, loss_down_threshold = 10**(-20), lr_down_threshold = 10**(-20)):
        super(Early_Stopping, self).__init__()
        self.loss_down_threshold = loss_down_threshold
        self.lr_down_threshold = lr_down_threshold

    def on_epoch_end(self, epoch, logs={}):

        self.loss = logs.get('loss')
        self.lr = float(K.get_value(self.model.optimizer.lr))

        if self.loss < self.loss_down_threshold:
            self.model.stop_training = True
            print("\nSTOP TRAINING - EPOCH {}: loss below".format(epoch+1), self.loss_down_threshold)

        if self.lr < self.lr_down_threshold:
            self.model.stop_training = True
            print("\nSTOP TRAINING - EPOCH {}: learning rate below".format(epoch+1), self.lr_down_threshold)

class History(Callback):

    def __init__(self):
        super(History, self).__init__()
        self.loss_history = list()
        self.lr_history = list()

    def on_epoch_end(self, epoch, logs={}):

        self.loss_history.append(logs.get('loss'))
        self.lr_history.append(float(K.get_value(self.model.optimizer.lr)))


class my_lr_criteria(Callback):
     
    def on_epoch_begin(self, epoch, logs={}):
        if epoch < 1:
            self.reference_weights = self.model.layers[1].get_weights()
        # if epoch >2:
        #    K.set_value(self.model.optimizer.lr, 2.)    
        return
       
    def on_epoch_end(self, epoch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        if epoch < 1:
            self.reference_loss = logs.get('loss')
            self.interrogant_weights = self.model.layers[1].get_weights()
            self.apcepted = False
        else:    
            print('lr', lr)
            print(logs.get('loss'))
            if logs.get('loss') < self.reference_loss:
                #we apcept!
                print('ACEPTAMOSSSSSSSSSSS***********')
                print()
                self.apcepted = True
                self.old_reference_loss = self.reference_loss
                K.set_value(self.model.optimizer.lr, lr * 1.1)    
                self.reference_weights = self.interrogant_weights
                self.interrogant_weights = self.model.layers[1].get_weights()
                self.reference_loss = logs.get('loss')
 
            elif logs.get('loss') > self.reference_loss:
                #we reject!
                print('REJECT****************************')
                print()
                self.apcepted = False
                K.set_value(self.model.optimizer.lr, lr * 0.85)
                self.model.layers[1].set_weights(self.reference_weights)
               
        if (self.apcepted and abs(abs(self.reference_loss) - abs(self.old_reference_loss)) < 10**(-4)) or lr < 1e-4:
                self.model.stop_training = True
                self.model.layers[1].set_weights(self.reference_weights)
                print('MY EARLY STOP')
                print((self.model.optimizer.lr).dtype)