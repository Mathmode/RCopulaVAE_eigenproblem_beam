# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:07:26 2020

@author: 109457
"""
import numpy as np
import pandas
import seaborn
import scipy
import sklearn
import matplotlib
from matplotlib import pyplot as plt

#Hacer función de configuración de gráficos
def plot_configuration():
    plt.rc('font', size = 18)          # controls default text sizes
    plt.rc('axes', titlesize = 18)     # fontsize of the axes title
    plt.rc('axes', labelsize = 18)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 18)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 18)   # legend fontsize
    plt.rc('figure', titlesize= 18)  # fontsize of the figure title

#Código gráficos
def plot_predicted_values_vs_ground_truth(gt_array, pred_array, title_label, filename):
    ymin = np.min((gt_array.min(),pred_array.min()))
    ymax = np.max((gt_array.max(), pred_array.max()))
    #min_val,max_val = ymin,ymax #for adaptive scaling
    min_val,max_val = -3,3 #for common scaling
    increment = max_val - min_val 
    limits = (min_val -0.05*increment, max_val + 0.05*increment)
    x,y = pandas.Series(gt_array,name = r'Ground Truth'),pandas.Series(pred_array, name = r'Predicted')
    g = seaborn.jointplot(x=x, y=y, kind = 'hex', color = '#1d6d68', joint_kws = {'gridsize':30,'bins':'log'},xlim=limits, ylim = limits, stat_func= None)
    #seaborn.regplot(pandas.Series(np.arange(xlim[0], xlim[1],0.01)),pandas.Series(numpy.arange(xlim[0],xlim[1],0.01)), ax = g.ax_joint, scatter = False)
    g.ax_joint.plot(np.linspace(limits[0], limits[1]), np.linspace(limits[0], limits[1]),'--r', linewidth=4)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(gt_array,pred_array)
    g.fig.suptitle(title_label, y = 0.99)
    textstr = '\n'.join((
        # r'$\mathrm{rms} = %.2f$'%(rms,),
        # '#data ='% (numpy.size(gt_array),),
        r'$\mathrm{r^2}=%.4f$'%(r_value**2,),
        # r'$\mathrm{p}=%.2f$'%(p_value,),
        ))
    
    #These are matplotlib.patch.Patch properties
    props = dict(boxstyle = 'round', alpha = 0.5, facecolor = 'none')
    # Place a textbox in upper left in axes coords
    g.ax_joint.text(0.05,0.95, textstr, transform  = g.ax_joint.transAxes, fontsize = 18, verticalalignment = 'top', bbox=props)
    plt.tight_layout()
    cbar_ax = g.fig.add_axes([1., 0.1, .075, .75])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)

    #plt.show()
    # if filename.endswith('.png'):
    g.fig.savefig(filename,dpi = 600)
    #     else:
    #g.fig.savefig(filename)
    # plt.close('all')
    # #close the pyplot sesion
    # plt.clf()
    # plt.cla()
    # plt.close()


def plot_crossplots(Xtrain_std,Xtest_std,train_predictions,test_predictions):
    seaborn.set(style = 'white', font_scale = 1.3)
    from MODULES.show_plots import plot_predicted_values_vs_ground_truth
    for i in range(Xtrain_std.shape[1]):
         plot_predicted_values_vs_ground_truth(Xtrain_std[:,i],train_predictions[:,i],'', 'Figures\corrplot_TrainS'+str(i+1))
    for i in range(Xtest_std.shape[1]):
        plot_predicted_values_vs_ground_truth(Xtest_std[:,i],test_predictions[:,i],'', 'Figures\corrplot_TestS'+str(i+1)+'_und') 

def plot_outliers(Train_rec_error,Test_rec_error,k):
    #Test_rec_errors = np.concatenate((Test_rec_error,Test_rec_errorD1)) 
    x = np.arange(len(Test_rec_error))
    Lim = np.percentile(Train_rec_error,100)
    col = np.where(Test_rec_error<Lim,'g','r')
    C_chart = plt.figure()
    for i in range(Test_rec_error.shape[0]):
        plt.scatter(x[i],Test_rec_error[i], s = 2, c=col[i])
    plt.ylabel('Reconstruction error')
    plt.xlabel('measurement')
    plt.plot(1)
    limit = np.ones(len(x))
    plt.plot(Lim*limit, color = '#0000FF', markersize = 8)
    C_chart.savefig('Figures\Controlchart.png', dpi = 500, bbox_inches='tight')
    plt.show()
    return Train_rec_error,Test_rec_error,Lim



def plot_outliers_dam(train_rec_error,test_rec_error, test_rec_errorD1,k,fac,percentile):
    # Train_rec_error = np.zeros(shape = (len(train_rec_error)-k,1))
    # for i in range(Train_rec_error.shape[0]):
    #     Train_rec_error[i,:] =  (np.sum(train_rec_error[i:(k+i),:]))/k
    # Test_rec_error = np.zeros(shape = (len(test_rec_error)-k,1))
    # for i in range(Test_rec_error.shape[0]):
    #     Test_rec_error[i,:] =  (np.sum(test_rec_error[i:(k+i),:]))/k

    # Test_rec_errorD1 = np.zeros(shape = (len(test_rec_errorD1)-k,1))
    # for i in range(Test_rec_errorD1.shape[0]):
    #     Test_rec_errorD1[i,:] =  (np.sum(test_rec_errorD1[i:(k+i),:]))/k
    Train_rec_error = train_rec_error
    Test_rec_error = test_rec_error
    Test_rec_errorD1 = test_rec_errorD1
    Test_rec_errors = np.concatenate((Test_rec_error,Test_rec_errorD1)) 
    #Test_rec_errors = Test_rec_error
    x = np.arange(len(Test_rec_errors))
    Lim = np.percentile(Train_rec_error,percentile)
    col = np.where(Test_rec_errors<Lim,'g','r')
    C_chart = plt.figure()
    for i in range( Test_rec_errors.shape[0]):
        plt.scatter(x[i], Test_rec_errors[i], s = 2, c=col[i])
    plt.ylabel('Damage indicator \u03C1 ')
    plt.xlabel('measurement')
    text_ypos = Lim+0.26
    C_chart.text(0.55,text_ypos, '\u03B1 = '+str(round(Lim,2)), color = '#0000FF')
    #C_chart.text(0.15,0.8, 'Damage level: '+str(round(100*(1-fac)))+'%', color = 'black')
    C_chart.text(0.15,0.8, 'No damaged data available yet!', color = 'black')
    #plt.plot(1)
    limit = np.ones(len(x))
    plt.plot(Lim*limit, color = '#0000FF', markersize = 8)
    C_chart.savefig('Figures\Controlchart.png', dpi = 500, bbox_inches='tight')
    plt.show()
    return Train_rec_error,Test_rec_error, Test_rec_errorD1, Test_rec_errors,Lim


def plot_loss_evolution(history):
    loss_plot  = plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(np.log10(loss),color = '#072FD1',)
    plt.plot(np.log10(val_loss), color = 'red')
    #plt.title('model loss')
    plt.ylabel('log(loss)')
    plt.xlabel('epoch')
    xmin, xmax = plt.xlim()
    # ymin, ymax = plt.ylim()
    ymin,ymax = -2, 0.0
    scale_factor = 1
    plt.xlim(xmin *scale_factor, xmax * scale_factor)
    plt.ylim(ymin * scale_factor, ymax * scale_factor)
    plt.legend(['train', 'validation'], loc='upper right', fontsize = 18)
    loss_plot.savefig('Figures\Model_loss_Porto',dpi = 500, bbox_inches='tight')
    plt.show()


def plot_histograms(Train_rec_error, percentile):
    Lim = np.percentile(Train_rec_error,percentile)
    Histo = plt.figure()
    plt.hist(Train_rec_error, bins = 16, color = 'skyblue', alpha=0.5, histtype='bar', ec='black')
    plt.axvline(x = Lim, ymax = 0.55, color = "red",linestyle='--', linewidth = '3')
    plt.axvline(x = Lim, ymin = 0.7, color = "red",linestyle='--', linewidth = '3')
    plt.xlabel('training reconstruction error')
    plt.ylabel('frequency')
    Histo.text(0.405, 0.58, 'p-99 = '+str(round(Lim,2)), color = 'red')
    Histo.savefig('Figures\Train_hist_Porto',dpi = 500, bbox_inches='tight')
    plt.show()
#def print_file(path,name,data,header):
    