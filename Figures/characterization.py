# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats.mstats import gmean
import scipy.optimize as opt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
from skimage import io


# %%
p = Path('D:/qs-analysis/compiled-data')

files = list(p.glob('**/*.pkl'))
df = pd.DataFrame(columns=['FITC-A','AHL','IPTG','Family','Replicate'])
for f in files:
    family = str(f.parent).split('\\')[-1]
    tmp = pd.read_pickle(f)
    df = df.append(tmp,ignore_index=True)





# %%
# Subtract WT background from samples
#data = data[data['FITC-A'] >= 0]
def clean_df(df):
    data = df
    data['FITC-A'] = data['FITC-A'] - data[data.Family=='WT']['FITC-A'].median()
    data = data[data.Family != 'WT']

    data.IPTG = (data.IPTG * 1e3)
    data.IPTG = data.IPTG.astype('int16')

    gb = data.groupby(['AHL','IPTG','Family'])
    
    lg = gb.AHL.transform(np.log10).sort_values()
    lg[lg == -np.inf] = 0
    lg = lg.astype('int32')
    lg.name = 'logAHL'
    return pd.concat([data,lg],axis=1)
Data = clean_df(df)


mfi = Data.groupby(['IPTG','Family','logAHL','Replicate']).median().reset_index()
mfi['R'] = mfi.groupby(['Family','IPTG','Replicate'])['FITC-A'].transform(lambda x: x.max()/x.min())





# %%
def make_heatmap(median_fluorescence,Family,ax=None, **kwargs):
    """
    This is a function to generate heatmaps
    
    Args:
        median_fluorescence (pd.DataFrame)
        DataFrame of filtered single family Lux/Las/Tra flow cytometry median fluorescence intensity (mfi)
        
        standard_deviation (pd.DataFrame)
        DataFrame of filtered single family Lux/Las/Tra flow cytometry standard deviation of fluorescence intensity (mfi)
        
        ax (plt.ax)
        Axis to plot data on
        
        **kwargs
        Addition arguments to pass to sns.heatmap
    """
    # sns.set_context('paper')
    ax = ax or plt.gca()
    mfi = median_fluorescence[median_fluorescence.Family ==Family]
    # normalized
    norm = mfi[mfi.logAHL != 0].copy()
    norm['FITC-A']=norm['FITC-A']/norm['FITC-A'].max()*100

    piv = norm.pivot_table(index='IPTG', columns='logAHL', values='FITC-A')
    g = sns.heatmap(piv, vmin=0,ax=ax,**kwargs)
    cbar = g.collections[0].colorbar
    cbar.set_ticks([0, 25, 50, 75,100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cbar.ax.set_ylabel('Relative Expression Level')
    ax.set_xlabel(r'AHL ($10^x$ M)')
    ax.set_ylabel(r'IPTG ($\mu M$)')
    #f.tight_layout()
    return ax


# %%
def get_hill_params(median_fluorescence,Family):
    """
    This is a function to generate Hill plots
    
    Args:
        median_fluorescence (pd.DataFrame)
            DataFrame of filtered single family Lux/Las/Tra flow cytometry median fluorescence intensity (mfi)
        Family (str)
            Name of AHL family to fit

    """
    mfi = median_fluorescence[median_fluorescence.Family ==Family]
    fits = pd.DataFrame(columns=['Slope','EC50','Min','Max','IPTG','Family'])
    #data = Data[Data.Family =='LuxR']
    #cmap = matplotlib.cm.get_cmap(colormap)
    #colors = cmap(np.linspace(0,1,len(IPTG_range)+1))
    for replicate in mfi.Replicate.unique():
        for iptg in mfi.IPTG.unique():
            median = mfi[(mfi.IPTG==iptg) & (mfi.Replicate == replicate)]
            max_ = median['FITC-A'].max()
            min_ = median['FITC-A'].min()
            median = mfi[(mfi.IPTG==iptg) & (mfi.Replicate == replicate) & (mfi.logAHL != 0)]
            #print(replicate,iptg)
            def func(x, a, b):
                return (max_  - min_) / (1 + 10**(a * (np.log10(b)-np.log10(x)))) + min_
            (a_, b_), _ = opt.curve_fit(func, median.AHL, median['FITC-A'],p0=[1,1e-9],maxfev=10000000)
            fits=fits.append(pd.DataFrame([[a_,b_,min_,max_,iptg,Family]],columns=['Slope','EC50','Min','Max','IPTG','Family']),ignore_index=True)

    return fits


# %%
def plot_hill(median_fluorescence,IPTG_range,Family,fits,colormap,ax=None,  **kwargs):
    """
    This is a function to generate Hill plots
    
    Args:
        median_fluorescence (pd.DataFrame)
        DataFrame of filtered single family Lux/Las/Tra flow cytometry median fluorescence intensity (mfi)
        
        standard_deviation (pd.DataFrame)
        DataFrame of filtered single family Lux/Las/Tra flow cytometry standard deviation of fluorescence intensity (mfi)
        
        IPTG_range (list)
        List of IPTG values to plot
        
        colormap (str)
        Name of colormap to use
        
        ax (plt.ax)
        Axis to plot data on
        
        **kwargs
        Addition arguments to pass to sns.heatmap
    """
    # sns.set_context('talk')
    ax = ax or plt.gca()
    lines = []  
    # fits = pd.DataFrame(columns=['AHL','FITC-A','IPTG'])
    cmap = matplotlib.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0,1,len(IPTG_range)+1))
    mfi = median_fluorescence[median_fluorescence.Family ==Family]
    mean = mfi.groupby(['IPTG','logAHL']).mean().reset_index()[['IPTG','logAHL','AHL','FITC-A']]
    std = mfi.groupby(['IPTG','logAHL']).std().reset_index()[['IPTG','logAHL','AHL','FITC-A']]
    for i,iptg in enumerate(IPTG_range):
        mn = mean[(mean.IPTG==iptg) & (mean.logAHL != 0)]
        sd = std[(std.IPTG==iptg) & (std.logAHL != 0)]
        fit = fits[fits.IPTG==iptg].mean()
        ax.errorbar(x=mn.logAHL.values,y=mn['FITC-A'].values,yerr=sd['FITC-A'].values,color=colors[i],capsize=5,ls='',marker='o')
        def func(x, slope, ec50,min_,max_):
            return (max_ -min_) / (1 + 10**(slope * (np.log10(ec50)-np.log10(x)))) + min_
        
        n = 1e6
        x = np.linspace(mn.AHL.min(), mn.AHL.max(), int(n))
        y_fit = func(x, fit['Slope'], fit['EC50'],fit['Min'],fit['Max'])
        # fits=fits.append(pd.DataFrame(np.concatenate([[x], [y_fit],[np.ones(len(x))*iptg]]).T,columns=['AHL','FITC-A','IPTG']),ignore_index=True)
        lines2, =ax.plot(np.log10(x), y_fit, '-',color = colors[i],label='IPTG {:.1f} fit'.format(iptg))
        lines += ax.plot(mn.logAHL.values, mn['FITC-A'].values, 'o',color = colors[i],label=r'IPTG {} $\mu M$'.format(iptg))

    labels = [l.get_label() for l in lines]
    #labels = ax.get_labels()
    ax.legend(handles=lines,labels=labels,frameon=False)
    ax.set_xlabel(r'AHL ($10^x$ M)')
    ax.set_ylabel(r'mNG Intensity (au)')
    sns.despine()
    return ax


# %%
def plot_hill_params(mfi,Family,colormap,grid,**kwargs):

    gs1 = grid.subgridspec(3, 1)
    axes=gs1.subplots()
    median_fluorescence = mfi[mfi.Family==Family]
    fit = get_hill_params(median_fluorescence,Family)
    #gs01 = grid.subgridspec(3, 1)
    #inner_grid = gridspec.GridSpecFromSubplotSpec(
    #  3, 1, subplot_spec=grid)
    #ax1 = figure.add_subplot(inner_grid[0,:])
   # ax2 = figure.add_subplot(inner_grid[1,:])
    #ax3 = figure.add_subplot(inner_grid[2,:])
    sns.barplot(x='IPTG',y='R',data=median_fluorescence,palette=colormap,ax=axes[0])

    axes[0].set_ylabel(r'Induction ratio ($\frac{mNG_{ON}}{mNG_{OFF}}$)')
    sns.barplot(x='IPTG',y='EC50',data=fit,ax=axes[1],ci='sd',palette=colormap)
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$EC_{50}$')
    sns.barplot(x='IPTG',y='Slope',data=fit,ax=axes[2],ci='sd',palette=colormap)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[2].set_xlabel(r'IPTG ($\mu M$)')
    sns.despine()
    return 



# %%
# fig = plt.figure(figsize=(16,9))
# gs = gridspec.GridSpec(3, 4, figure=fig)
# axs1 = gs.subplots()
# iptg = 5

def ridge_plot(Data,family,iptg,colormap,grid):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    temp = Data[(Data.Family==family) & (Data.IPTG ==iptg)]
    gs1 = grid.subgridspec(len(temp.AHL.unique()), 1,hspace=-.5)
    axes=gs1.subplots()
    cmap = matplotlib.cm.get_cmap(colormap)
    colors = cmap(range(len(temp.AHL.unique())))
    for color,c,ax in zip(colors,temp.AHL.unique(),axes.ravel()):
        temp2 = temp[temp.AHL == c]
        sns.kdeplot(temp[temp.AHL == c]['FITC-A'],clip_on=True, fill=True, alpha=1, lw=1,bw_adjust=3,ax=ax,color=color) #log_scale=True,
        sns.kdeplot(temp[temp.AHL == c]['FITC-A'],clip_on=True, color='w', alpha=1, lw=2,bw_adjust=3,ax=ax)
        ax.axhline(y=0, lw=2,c=color)
    spines = ["top","right","left","bottom"]
    for j,ax in enumerate(axes.ravel(),1):
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.set_xlim(1e1,2e5)
        ax.set_xscale('log')
        for s in spines:
            ax.spines[s].set_visible(False)
        if j == len(temp.AHL.unique()):
            ax.set_xlabel('FITC-A')
            
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')


# ridge_plot(Data,'LuxR',5,'Blues',grid=gs[0,0])
# ridge_plot(Data,'LasR',5,'Reds',grid=gs[1,0])
# ridge_plot(Data,'TraR',5,'Greens',grid=gs[2,0])


# %%



# %%
fig = plt.figure(constrained_layout=True, figsize=(6.5,10))
gs0 = gridspec.GridSpec(5, 3, figure=fig)
sns.set_context('paper')
sns.set_style('white')
spines = ["top","right","left","bottom"]
# Initialize plot axes
axes = gs0.subplots()

# Diagrams

axs = axes.ravel()
families = ['LuxR','LasR','TraR']
for i,family in enumerate(families):
    img = io.imread(f'QS Families_{family}.png')
    axs[i].imshow(img)
    axs[i].axis('off')
# ridge plots
ax1 = fig.add_subplot(gs0[1, 0])
ax2 = fig.add_subplot(gs0[1, 1])
ax3 = fig.add_subplot(gs0[1, 2])

# Hill plots
ax4 = fig.add_subplot(gs0[2, 0])
ax5 = fig.add_subplot(gs0[2, 1])
ax6 = fig.add_subplot(gs0[2, 2])

# Heatmaps
ax7 = fig.add_subplot(gs0[3, 0])
ax8 = fig.add_subplot(gs0[3, 1])
ax9 = fig.add_subplot(gs0[3, 2])

# Hill Params
#ax10 = fig.add_subplot(gs0[0, 3])
#ax11 = fig.add_subplot(gs0[1, 3])
#ax12 = fig.add_subplot(gs0[2, 3])

# TODO fix this
#plot_hill(mfi,[0,5,500],'LuxR',fits,'Blues_r',ax=ax)
plot_hill(mfi,std,[0,5,500],'LuxR','Blues_r',ax4)
plot_hill(mfi,std,[0,5,500],'LasR','Reds_r',ax5)
plot_hill(mfi,std,[0,5,500],'TraR','Greens_r',ax6)

make_heatmap(mfi,'LuxR',ax7,cmap='Blues_r')
make_heatmap(mfi,'LasR',ax8,cmap= 'Reds_r')
make_heatmap(mfi,'TraR',ax9,cmap='Greens_r')

plot_hill_params(mfi,'LuxR',colormap='Blues_r',grid=gs0[4, 0])
plot_hill_params(mfi,'LasR',colormap='Reds_r',grid=gs0[4, 1])
plot_hill_params(mfi,'TraR',colormap='Greens_r',grid=gs0[4, 2])

ridge_plot(Data,'LuxR',5,'Blues',grid=gs0[1,0])
ridge_plot(Data,'LasR',5,'Reds',grid=gs0[1,1])
ridge_plot(Data,'TraR',5,'Greens',grid=gs0[1,2])

for ax in [ax1,ax2,ax3]:
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for s in spines:
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig('SE Characterization.png',dpi=600)


# %%


# %%



