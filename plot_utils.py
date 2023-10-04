import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf
from neuromaps.datasets import fetch_fslr
from neuromaps.parcellate import Parcellater

def custom_surf_plot(data, density='32k', cmap='coolwarm', dpi=250, cbar_label=None,
                   atlas=None, vmin=None, vmax=None):
    """
    Plot nice plots in fsLR space.
    
    Parameters
    ----------
    data : array_like or tuple
        ROI-wise or vertex-wise data. If tuple, assumes (left, right) hemisphere.
    density : str
        Density of surface plot, can be '8k', '32k' or '164k'.
    cmap : str
        Colormap.
    dpi : int
        Resolution of plot.
    cbar_label: str, optional
        Colorbar label.
    atlas : Path, optional
        Path to an atlas in .dlabel.nii format.
    vmin/vmax : int, optional
        Minimun/ maximum value in the plot.
    """
    if atlas is not None:
        surf_masker = Parcellater(atlas, space)
        data = surf_masker.inverse_transform(data)
        l_data, r_data = data[0].agg_data(), data[1].agg_data()
    else:
        if not isinstance(data, tuple):
            raise ValueError("Data input must be tuple-of-arrays. Alternatively provide 'atlas' for ROI data.")
        l_data, r_data = data[0], data[1]
        
    if None in (vmin, vmax):
        # Handle NaNs in left hemisphere data
        l_min, l_max = np.nanmin(l_data), np.nanmax(l_data)
        l_data = np.nan_to_num(l_data, nan=l_min)
        
        # Handle NaNs in right hemisphere data
        r_min, r_max = np.nanmin(r_data), np.nanmax(r_data)
        r_data = np.nan_to_num(r_data, nan=r_min)
        
        # min/max values in the data
        vmin = np.min(l_min, r_min)
        vmax = np.max(l_max, r_max)

    # %% Plot both hemispheres
    fig, ax = plt.subplots(nrows=1,ncols=4,subplot_kw={'projection': '3d'}, figsize=(12, 4), dpi=dpi)
    plot_surf(lh, l_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral',
            colorbar=False, axes=ax.flat[0])
    plot_surf(lh, l_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial',
            colorbar=False, axes=ax.flat[1])

    plot_surf(rh, r_data, threshold=-1e-14, cmap=cmap, alpha=1, view='lateral',
            colorbar=False, axes=ax.flat[2])
    p = plot_surf(rh, r_data, threshold=-1e-14, cmap=cmap, alpha=1, view='medial',
                colorbar=True, cbar_tick_format='%.0f', axes=ax.flat[3])

    p.axes[-1].set_ylabel(cbar_label, fontsize=10, labelpad=0.5)
    p.axes[-1].set_yticks([vmin, vmax])
    p.axes[-1].set_yticklabels(['min', 'max'])
    p.axes[-1].tick_params(labelsize=7, width=0, pad=0.1)
    plt.subplots_adjust(wspace=-0.1)
    p.axes[-1].set_position(p.axes[-1].get_position().translated(0.08, 0))