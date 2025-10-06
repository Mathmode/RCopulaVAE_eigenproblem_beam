# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:52:37 2020

@author: 109457
"""
import numpy as np
import pandas
import seaborn
import keras as K
import scipy
import sklearn
import os 
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from MODULES.TRAINING.deterministic_models import My_Inverse_withPhysics, My_Inverse


#Graph configuration function (font sizes)
def plot_configuration():
    plt.rc('font', size = 16)          # controls default text sizes
    plt.rc('axes', titlesize = 16)     # fontsize of the axes title
    plt.rc('axes', labelsize = 16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 16)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 16)   # legend fontsize
    plt.rc('figure', titlesize= 16)   # fontsize of the figure title


def plot_loss_evolution(history, output_folder_path):
    loss_plot  = plt.figure()
    loss = history.history['loss']
    # loss_loc = history.history['custom_loss_Location']
    # loss_sev = history.history['custom_loss_Severity']
    val_loss = history.history['val_loss']
    plt.plot(val_loss,color = '#072FD1',)
    plt.plot(loss,color = 'red')
    # plt.plot(loss_loc,color = 'red',)
    # plt.plot(loss_sev,color = 'green',)
    #plt.title('model loss')
    plt.ylabel('$\mathcal{L}_{total}$')
    plt.xlabel('epoch')
    xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()
    # ymin,ymax = 0.001, 0.1
    scale_factor = 1
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin *scale_factor, xmax * scale_factor)
    # plt.ylim(ymin * scale_factor, ymax * scale_factor)
    plt.legend(['Validation', 'Training'], loc='upper right', fontsize = 14)
    plt.show()
    loss_plot.savefig(os.path.join(output_folder_path, "Loss_trainval.png"),dpi = 500, bbox_inches='tight')

    
def plot_loss_terms(model_history, folder_path=None):
    # Extract the loss terms from the history
    Freqs_loss = model_history['freqs_loss']
    MAC_loss = model_history['mac_modes_loss']
    Regularizer_loss = np.array(model_history['alpha_regularizer'])*-1.
    total_loss = model_history['loss']
    # Create a figure
    plt.figure(figsize=(10, 6))
    # Plot each loss term with a different style
    plt.plot(Freqs_loss, label='${\gamma}\mathcal{L}_{freq}$', color='blue', linestyle=':', linewidth=2.)
    plt.plot(MAC_loss, label='$\mathcal{L}_{MAC}$', color='green', linestyle='-.', linewidth=2.)
    plt.plot(Regularizer_loss, label=r'$-{\varepsilon}\mathcal{R}_{\alpha}$', color='orange', linestyle='--', linewidth=2.)
    plt.plot(total_loss, label='Total Loss', color='red', linestyle='-', linewidth=3)
    # Add titles and labels
    # plt.title('Loss Terms Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss value', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    # Add a legend
    plt.legend(loc='upper right', fontsize=12)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # plt.show()
    plt.savefig(os.path.join(folder_path, 'loss_terms_plot.png'),dpi = 500, bbox_inches='tight')
    # plt.close()




def show_predicted_factors(alpha_true, alpha_pred):
    low = 0
    high = alpha_true.shape[0] #Maximum position value
    k = 100
    # Create an array of 20 random integers between 1 and 100
    random_integers = np.random.randint(low, high, size=k)
    
    for i in range(k):
        pos = random_integers[i]
        print(alpha_pred[pos,:])
        print(alpha_true[pos,:])
        print('************************************************')

    
def plot_alpha_crossplots(true_alphas, pred_alphas, folder_path):
    
    # Assuming pred_alphas and true_alphas are already defined arrays
    # pred_alphas  = np.min(alphapred, axis=1)
    # true_alphas = np.min(alpha_factors_true_test, axis=1)
    
    # Calculate the R² score
    r2 = r2_score(true_alphas, pred_alphas)
    
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    
    # Plot the true values against the predicted values
    plt.scatter(true_alphas, pred_alphas, color='blue', alpha=0.6, edgecolors='k', label='Predicted vs True')
    
    # Add a 1:1 reference line (ideal prediction line)
    min_val = min(min(true_alphas), min(pred_alphas))
    max_val = max(max(true_alphas), max(pred_alphas))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='1:1 Line')
    
    # Add titles and labels
    # plt.title('Crossplot of True vs Predicted Alphas', fontsize=16)
    plt.xlabel('True Alpha Values', fontsize=14)
    plt.ylabel('Predicted Alpha Values', fontsize=14)
    
    # Add a legend
    plt.legend(loc='lower right', fontsize=12)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Display the R² score in a small text box
    textstr = f'R² = {r2:.2f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Show the plot
    # plt.show()   
    plt.savefig(os.path.join(folder_path, 'crossplot_minalpha.png'),dpi = 500, bbox_inches='tight')
    # plt.close()
    

    
    
    
    
    









































#cumulated damage indicator with k 
def cumulated_errors(train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam, k):
    Train_rec_error = np.zeros(shape = (len(train_rec_error)-k,1))
    for i in range(Train_rec_error.shape[0]):
        Train_rec_error[i,:] =  (np.sum(train_rec_error[i:(k+i),:]))/k
    
    Val_rec_error = np.zeros(shape = (len(val_rec_error)-k,1))
    for i in range(Val_rec_error.shape[0]):
        Val_rec_error[i,:] =  (np.sum(val_rec_error[i:(k+i),:]))/k
    
    Test_rec_error = np.zeros(shape = (len(test_rec_error)-k,1))
    for i in range(Test_rec_error.shape[0]):
        Test_rec_error[i,:] =  (np.sum(test_rec_error[i:(k+i),:]))/k
    
    Test_rec_error_dam= np.zeros(shape = (len(test_rec_error_dam)-k,1))
    for i in range(Test_rec_error_dam.shape[0]):
        Test_rec_error_dam[i,:] =  (np.sum(test_rec_error_dam[i:(k+i),:]))/k
    
    Test_rec_errors = np.concatenate((Test_rec_error,Test_rec_error_dam))
    return Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors

    
    
    





# def plot_loss_evolution(history,f_name):
#     loss_plot  = plt.figure()
#     loss = history.history['loss']
#     # loss_loc = history.history['custom_loss_Location']
#     # loss_sev = history.history['custom_loss_Severity']
#     val_loss = history.history['val_loss']
#     plt.plot(val_loss,color = '#072FD1',)
#     plt.plot(loss,color = 'red')
#     # plt.plot(loss_loc,color = 'red',)
#     # plt.plot(loss_sev,color = 'green',)
#     #plt.title('model loss')
#     plt.ylabel('$loss_{S}$')
#     plt.xlabel('epoch')
#     xmin, xmax = plt.xlim()
#     # ymin, ymax = plt.ylim()
#     ymin,ymax = 0.001, 0.1
#     scale_factor = 1
#     plt.yscale('log')
#     plt.xlim(xmin *scale_factor, xmax * scale_factor)
#     plt.ylim(ymin * scale_factor, ymax * scale_factor)
#     plt.legend(['Validation', 'Training'], loc='upper right', fontsize = 14)
#     loss_plot.savefig(os.path.join("Figures",str(f_name)+"_loss_trainval"),dpi = 500, bbox_inches='tight')
#     plt.show()


def plot_histogram(Train_rec_error, Lim, percentile):
    Histo = plt.figure()
    plt.hist(Train_rec_error, bins = 16, color = 'skyblue', alpha= 0.5, histtype='bar', ec='black')
    plt.axvline(x = Lim, ymax = 0.55, color = "red",linestyle='--', linewidth = '3')
    plt.axvline(x = Lim, ymin = 0.7, color = "red",linestyle='--', linewidth = '3')
    plt.xlabel('training reconstruction error')
    plt.ylabel('frequency')
    Histo.text(0.415, 0.58, 'p-99 = '+str(round(Lim,2)), color = 'red')
    Histo.savefig(os.path.join("Figures","Train_hist_Porto"),dpi = 500, bbox_inches='tight')
    plt.show()
    return Lim

def calculate_errors(Xstd, predictions):
    rec_error = np.zeros(shape = [ Xstd.shape[0],1])
    for i in range(Xstd.shape[0]):
        rec_error[i] = np.sum((Xstd[i] - predictions[i])**2)
    return rec_error


def calculate_metrics(Test_rec_error, Test_rec_error_dam, Lim):
    FP = len(np.where(Test_rec_error > Lim)[0])
    FN = len(np.where(Test_rec_error_dam < Lim)[0])
    TP = len(np.where(Test_rec_error_dam>Lim)[0])
    TN = len(np.where(Test_rec_error<Lim)[0])
    
    accuracy = (TP+TN)/(2*len(Test_rec_error))*100
    precision = TP/(TP+FP)*100
    recall = TP/(TP+FN)*100
    f1_score = 2*(precision*recall)/(precision + recall)
    print(accuracy,precision, recall, f1_score)
    return accuracy, precision, recall, f1_score



def plot_healthy_measurements(Healthy_preds, fig_name):
    x = np.linspace(1,Healthy_preds.shape[0],Healthy_preds.shape[0])
    plt.plot(x,Healthy_preds, 'o', color='blue', markersize = 1.5)
    Ground_truth = np.zeros(shape = Healthy_preds.shape[0])
    Ground_truth = Ground_truth +0.05
    # plt.plot(x, Ground_truth, '+', color='black', markersize = 2)
    plt.plot(0.0)
    plt.plot(0.5)
    plt.xlabel('Observation')
    plt.ylabel(' Severity')
    plt.savefig(os.path.join("Figures", "Testing_Figures_Jul022",fig_name),dpi = 500, bbox_inches='tight')


def plot_synthetic_testing_sev(ground_truth, predictions, fig_name):
    Sev_test = plt.figure()
    x = np.linspace(0,0.5,100)
    y = x
    plt.plot(x,y ,'--', color='red')
    plt.scatter(ground_truth[0:400,0], predictions[0:400,0],s =15, color = 'blue')
    plt.scatter(ground_truth[400:800,0], predictions[400:800,0],s =15, color = 'purple')
    plt.scatter(ground_truth[800:1200,0], predictions[800:1200,0],s =15, color = 'deepskyblue')
    # plt.scatter(ground_truth[1200:1600,0], predictions[1200:1600,0],s =15, color = 'limegreen')
    # plt.scatter(ground_truth[1600:,0], predictions[1600:,0],s =15, color = 'gold')

    plt.xlabel('Ground truth ')
    plt.ylabel(' Predicted severity')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ground_truth[:,0],predictions[:,0])
    ax = Sev_test.add_subplot()
    # ax.text(0.0,0.45, r'$r^{2} =$', fontsize =15)
    # ax.text(0.06, 0.45, str(np.round(r_value,2)), fontsize=15)
    # ax.text(0.39, 0.025, r'$r^{2} =$', fontsize =15)
    # ax.text(0.44, 0.025, str(0.3713), fontsize=15)
    ax.text(0.0025, 0.47, r'$r^{2} =$', fontsize =15)
    ax.text(0.065, 0.47, str(0.3713), fontsize=15)
    Sev_test.savefig(os.path.join("Figures","Testing_Figures_Jul022",fig_name),dpi = 500, bbox_inches='tight')



def plot_synthetic_testing_loc(ground_truth, predictions, fig_name):
    Loc_test = plt.figure()
    x = np.linspace(1,8,100)
    y = x
    plt.plot(x,y ,'--', color='red')
    plt.scatter(ground_truth[0:400,0], predictions[0:400,0],s =15, color = 'blue')
    plt.scatter(ground_truth[400:800,0], predictions[400:800,0],s =15, color = 'purple')
    plt.scatter(ground_truth[800:1200,0], predictions[800:1200,0],s =15, color = 'deepskyblue')
    plt.scatter(ground_truth[1200:1600,0], predictions[1200:1600,0],s =15, color = 'limegreen')
    plt.scatter(ground_truth[1600:,0], predictions[1600:,0],s =15, color = 'gold')


    # x = np.array([1,2,3,4,5,6,7,8])
    # y = x
    # plt.scatter(ground_truth[:,0], predictions[:,0], s = 15, color = 'blue')
    # plt.scatter(x,y, s = 40, marker = 'X', c='red')
    plt.xlabel('Ground truth ')
    plt.ylabel(' Predicted location')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ground_truth[:,0],predictions[:,0])
    ax = Loc_test.add_subplot()

    # ax.text(1.0, 7.5, r'$r^{2} =$', fontsize =15)
    # ax.text(1.8, 7.5, str(0.8864), fontsize=15)
    ax.text(6.4, 1.0, r'$r^{2} =$', fontsize =15)
    ax.text(7.14, 1.0, str(0.6187), fontsize=15)
    Loc_test.savefig(os.path.join("Figures","Testing_Figures_Jul022",fig_name),dpi = 500, bbox_inches='tight')





# PLOT DIFFERENT FITNESS SEVERITY CORRELATION BY ZONE
# Z1  = Data[Data[:,46] ==1]
# Z1  = Z1[Z1[:,48]<0.5]
# plt.plot(Z1[:,47],Z1[:,48],'.')
# Z2  = Data[Data[:,46] ==2]
# plt.plot(Z2[:,47],Z2[:,48],'.')
# Z3  = Data[Data[:,46] ==3]
# plt.plot(Z3[:,47],Z3[:,48],'.')
# Z4  = Data[Data[:,46] ==4]
# plt.plot(Z4[:,47],Z4[:,48],'.')
# Z5  = Data[Data[:,46] ==5]
# Z5[:,48] = np.log(Z5[:,48])
# Z5[:,47] = Z5[:,47]**2
# plt.plot(Z1[:,47],Z1[:,48],'.')
# x,y = pandas.Series(Z5[:,47],name = r'Severity'),pandas.Series(Z5[:,48], name = r'Fitness')
# seaborn.jointplot(x= x, y = y, kind = 'hex', joint_kws = {'gridsize':30,'bins':'log'})

#plot time domain vibration signals for graphics 
# dt = 1/4800  inverse of the frequency 
# t = np.zeros(Data[:,0].shape)
# for i in range(t.shape[0]):
#     t[i] = (i)*dt
# plt.plot(t[110000:135000], Data[110000:135000,0],color = '#0000FF')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration ($\mathrm{m/s^2}$)')


    

