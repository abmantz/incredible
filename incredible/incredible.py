import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st

def whist(x, smooth=True, kde_n=512, kde_range=None, bins='auto', plot=None,
              kde_kwargs=None, hist_kwargs=None, **kwargs):
    """
    Turn an array of samples, x, into an estimate of probability density at a
    discrete set of x values, possibly with some weights for the samples, and 
    possibly doing some smoothing.
    
    Return value is a dictionary with keys 'x' and 'density'.
    
    Calls scipy.stats.gaussian_kde if smooth=True, numpy.histogram otherwise.
    If smooth=False or None, does no smoothing.
    If smooth is a positive integer, do fixed-kernel Gaussian smoothing.

    Additional options:
     kde_n (kde) - number of points to evaluate at (linearly covers kde_range)
     kde_range (kde) - range of kde evaluation (defaults to range of x)
     bins (histogram) - number of bins to use or array of bin edges
     plot (either) - if not None, plot the thing to this matplotlib device
     kde_kwargs - dictionary of options to gaussian_kde
     hist_kwargs - dictionary of options to histogram
     **kwargs - additional options valid for EITHER gaussian_kde or histogram,
       especially `weights`
    """
    if kde_kwargs is None:
        kde_kwargs = {}
    if hist_kwargs is None:
        hist_kwargs = {}
    if plot is not None:
        plot.hist(x, bins=bins, density=True, fill=False, **kwargs);
    if smooth is True:
        if kde_range is None:
            kde_range = (x.min(), x.max())
        h = {'x': np.linspace(kde_range[0], kde_range[1], kde_n)}
        h['density'] = st.gaussian_kde(x, **kde_kwargs, **kwargs)(h['x'])
    else:
        hi = np.histogram(x, bins=bins, density=True, **hist_kwargs, **kwargs)
        nb = len(hi[0])
        h = {'x': 0.5*(hi[1][range(1,nb+1)]+hi[1][range(nb)]), 'density': hi[0]}
        if smooth is False or smooth is None or smooth == 0:
            pass
        else:
            h['density'] = gaussian_filter(h['density'], smooth, mode='constant', cval=0.0)
    if plot is not None:
        plot.plot(h['x'], h['density'], 'b-');
    return h

def ci1D_plot(h, ci, plot, plot_mode=True, plot_levels=True, plot_ci=True, fill_ci=True, pdf_kwargs=None, mode_kwargs=None, level_kwargs=None, ci_kwargs=None, fill_kwargs=None, fill_colors=None):
    """
    `h` is a dictionary with keys 'x' and 'density' (e.g. from `whist`).
    `ci` is output from whist_ci

    plot_mode, plot_levels, plot_ci, fill_ci - bells and whistles to include
    If `ci` has a 'color' entry, this will override `fill_colors` for shading
      of the intervals. This can be useful if there are multiply connected CI's,
      which is a pain for upstream programs to test for.
    """
    if pdf_kwargs is None:
        pdf_kwargs = {}
    if mode_kwargs is None:
        mode_kwargs = {}
    if level_kwargs is None:
        level_kwargs = {}
    if ci_kwargs is None:
        ci_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    if fill_ci:
        for i in range(len(ci['low'])-1, -1, -1):
            kw = {'color':str(ci['level'][i])}
            for k in fill_kwargs.keys():
                kw[k] = fill_kwargs[k]
            if fill_colors is not None:
                kw['color'] = fill_colors[i]
            try:
                kw['color'] = ci['color'][i]
            except KeyError:
                pass
            j = np.where(np.logical_and(h['x']>=ci['min'][i], h['x']<=ci['max'][i]))[0]
            plot.fill(np.concatenate(([ci['min'][i]], h['x'][j], [ci['max'][i]])), np.concatenate(([0.0], h['density'][j], [0.0])), **kw)
    kw = {'color':'k', 'marker':'', 'linestyle':'-'}
    for k in pdf_kwargs.keys():
        kw[k] = pdf_kwargs[k]
    plot.plot(h['x'], h['density'], **kw);
    if plot_mode:
        kw = {'color':'b', 'marker':'o', 'linestyle':''}
        for k in mode_kwargs.keys():
            kw[k] = mode_kwargs[k]
        plot.plot(ci['mode'], h['density'].max(), **kw);
    for i in range(len(ci['low'])):
        if plot_ci:
            kw = {'color':'b', 'marker':'', 'linestyle':'-'}
            for k in ci_kwargs.keys():
                kw[k] = ci_kwargs[k]
            plot.plot([ci['min'][i],ci['min'][i]], [0.0,ci['density'][i]], **kw);
            plot.plot([ci['max'][i],ci['max'][i]], [0.0,ci['density'][i]], **kw);
        if plot_levels:
            kw = {'color':'g', 'marker':'', 'linestyle':'--'}
            for k in level_kwargs.keys():
                kw[k] = level_kwargs[k]
            plot.axhline(ci['density'][i], **kw)

def whist_ci(h, sigmas=np.array([1.0,2.0]), prob=None, plot=None, **plot_kwargs):
    """
    Accept a dictionary with keys 'x' and 'density' (e.g. from `whist`).
    Entries in 'x' must be equally spaced.
    
    Return the mode of the PDF, along with the endpoints of the HPD interval(s) 
    containing probabilities in `prob`, or number of "sigmas" given in `sigmas`.
    
    plot - if not None, plot the thing to this matplotlib thingy
    **plot_kwargs - keywords to pass to ci1D_plot
    """
    if prob is None:
        prob = st.chi2.cdf(sigmas**2, df=1)
    mode = h['x'][np.argmax(h['density'])]
    imin = []
    imax = []
    iden = []
    ilev = []
    iprob = []
    theintervals = []
    o = np.argsort(-h['density']) # decreasing order
    k = -1
    for p in prob:
        k += 1
        for j in range(len(o)):
            if np.sum(h['density'][o[range(j)]]) / np.sum(h['density']) >= p: # NB failure if bins are not equal size
                reg = np.sort(o[range(j)])
                intervals = [[reg[0]]]
                if j > 0:
                    for i in range(1, len(reg)):
                        if reg[i] == reg[i-1]+1:
                            intervals[-1].append(reg[i])
                        else:
                            intervals.append([reg[i]])
                for i in range(len(intervals)):
                    imin.append(np.min(h['x'][intervals[i]]))
                    imax.append(np.max(h['x'][intervals[i]]))
                    iden.append(h['density'][o[j]])
                    ilev.append(p)
                    iprob.append( np.sum(h['density'][intervals[i]]) / np.sum(h['density']) )
                    theintervals.append(intervals[i])
                break
    imin = np.array(imin)
    imax = np.array(imax)
    ilev = np.array(ilev)
    iprob = np.array(iprob)
    iden = np.array(iden)
    ilow = imin - mode
    ihig = imax - mode
    icen = 0.5*(imin + imax)
    iwid = 0.5*(imax - imin)
    res =  {'mode':mode, 'level':ilev, 'prob':iprob, 'density':iden,
                'min':imin, 'max':imax, 'low':ilow, 'high':ihig, 'center':icen,
                'width':iwid}
    if plot is not None:
        ci1D_plot(h, res, plot, **plot_kwargs)
    return res


def whist2d(x, y, smooth=None, plot=None, **kwargs):
    """
    Two-dimensional version of `whist`.

    Returns dictionary with entries 'x' and 'y' (1D arrays) and 'z' (2D array).
    
    **kwargs - options to pass on to numpy.histogram2d, e.g.
     weights
     bins
     (density=True is passed by fiat)
    
    Additional options:
     smooth - width of Gaussian smoothing (NOT True/False)
     plot - if not None, plot 2D histogram to this matplotlib thingy
    """
    h, xbreaks, ybreaks = np.histogram2d(x, y, density=True, **kwargs)
    if smooth is not None:
        h = gaussian_filter(h, smooth, mode='constant', cval=0.0)
    nx = len(xbreaks) - 1
    ny = len(ybreaks) - 1
    # transpose z so that x and y are actually x and y
    h = h.T
    if plot is not None:
        plt.imshow(h, origin='lower')
    return {'x': 0.5*(xbreaks[range(1,nx+1)]+xbreaks[range(nx)]),
            'y': 0.5*(ybreaks[range(1,ny+1)]+ybreaks[range(ny)]),
            'z': h
           }

def _get_contour_verts(cn):
    '''
    Get coordinates of the lines drawn by pyplot.contour.
    https://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour

    WARNING: In practice, this appears to fail very badly if there are many contours.
    '''
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours

def whist2d_ci(h, sigmas=np.array([1.0,2.0]), prob=None, plot=None,
                   mode_fmt='bo', contour_color='k'):
    """
    Two dimension version of whist_ci.

    Accepts a dictionary with keys 'x', 'y' and 'z' (1D, 1D, 2D arrays).
    'x' and 'y' entries must be equally spaced.

    Returns the mode of the PDF, along with the contours of the HPD regions(s)
    containing probabilities in `prob`.
     'mode' - array (length 2)
     'levels' - array (probabilities equivalent to sigmas, or reproduces prob)
     'contours' - list of (for each level) list of (for each contours) arrays
                  with shape (:,2) storing the contour vertices

    plot - if not None, plot the thing to this matplotlib device
           if None, a plot will be created, because python
    Set `mode_fmt` to None to supress plotting of the mode

    WARNING: while the contour plot generated by this function should be
             accurate, the returned contours may be wrong if there are many
             or very complicated contours. Comparing them by overplotting the
             returned contours with `ci2D_plot` is recommended, if in doubt.
    """
    if plot is None:
        plot = plt.axes()
    if prob is None:
        prob = st.chi2.cdf(sigmas**2, df=1)
    imode = np.unravel_index(np.argmax(h['z']), h['z'].shape)
    mode = np.array([h['x'][imode[1]], h['y'][imode[0]]])
    o = np.argsort(-h['z'], None) # decreasing order
    k = -1
    level = np.zeros(len(prob))
    j = 0
    for p in prob:
        k += 1
        for j in range(j, len(o)):
            if np.sum(h['z'][np.unravel_index(o[range(j)], h['z'].shape)]) / np.sum(h['z']) >= p:
                level[k] = h['z'][np.unravel_index(o[j], h['z'].shape)]
                break
    contours = []
    for lev in level:
        contours.append( _get_contour_verts(plot.contour(h['x'], h['y'], h['z'], levels=[lev], colors=contour_color))[0] )
    if mode_fmt is not None:
        plot.plot(mode[0], mode[1], mode_fmt)
    return {'mode':mode, 'density':level, 'level':prob, 'contours':contours}

def ci2D_plot(contours, plot, transpose=False, outline=True, fill=False,
                  Line2D_kwargs=None, fill_kwargs=None):
    '''
    `contours` argument should be an entry from the 'contours' element returned
    from whist2d_ci (a list of n*2 arrays)

    `plot` is a matplotlib thingy on which to plot

    Everything else is hopefully self explanatory
    '''
    if Line2D_kwargs is None:
        Line2D_kwargs = {}
    if fill_kwargs is None:
        fill_kwargs = {}
    if transpose:
        i = 1
        j = 0
    else:
        i = 0
        j = 1
    lkw = {'linestyle':'-', 'color':'k'}
    for k in Line2D_kwargs.keys():
        lkw[k] = Line2D_kwargs[k]
    fkw = {'color':'0.5'}
    for k in fill_kwargs.keys():
        fkw[k] = fill_kwargs[k]
    for con in contours:
        if fill:
            plot.fill(con[:,i], con[:,j], **fkw);
        if outline:
            plot.plot(con[:,i], con[:,j], **lkw);


def whist_triangle(chain, smooth1D=True, smooth2D=None, whist_kwargs={},
                       whist2d_kwargs={}, ci_kwargs={}, **kwargs):
    """
    Compute 1D histograms for every column in `chain`, and 2D histograms for every pair of columns.
    Returns a list of lists with entries [0][0], [1][0], [1][1], [2][0][0], etc.
    "Diagonal" entries hold [h,ci], the outputs of `whist` and `whist_ci`.
    "Off-diagonal" entries hold the output of `whist2d_ci`.
    `smooth1D` is True or False; see `whist`.
    `smooth2D` is None or a positive real number; see `whist2d`.
    `ci_kwargs` are passed to both `whist_ci` and `whist2d_ci`.
    `kwargs` are passed to both `whist` and `whist2d`. Likely enties are `weights` and `bins`.
    """
    res = []
    for i in range(chain.shape[1]):
        res.append([])
        for j in range(i):
            h = whist2d(chain[:,j], chain[:,i], smooth=smooth2D, plot=None, **whist2d_kwargs, **kwargs)
            res[i].append( whist2d_ci(h, **ci_kwargs) )
        h = whist(chain[:,i], smooth=smooth1D, plot=None, **whist_kwargs, **kwargs)
        res[i].append( [h, whist_ci(h, **ci_kwargs)] )
    return res

# lots of pyplot formatting stuff copied from https://github.com/SebastianBocquet/pygtc
def whist_triangle_plot(tri, paramNames=None, axes=None, ranges=None, Pmin=0.0,
                        Pmax=None, density_label=None, width=7.0,
                        grid=False, label_kwargs=None, tick_kwargs=None,
                        skip1D = False, skip2D=False,
                        linecolor1D='k', fill1D=False, fillcolors1D=None,
            linecolor2D='k', linestyle2D='-', fill2D=True, fillcolors2D=None,
                        ci1D_kwargs=None, ci2D_kwargs=None):
    """
    Make a plot from the complicated thing returned by `whist_triangle`.

    If `axes` is None, returns a pyplot Figure and Axes pair for the plot.
    To overplot a second (or third, or ...) triangle, pass the latter as `axes`.

    Defaults are to plot just 1D PDFs on the diagonal, and filled contours in
    the lower triangle. Override all kinds of stuff with the various kwargs.

    If fill colors are not passed, they will be set to grayscale, scaling with
    the level of the corresponding credible region.

    Specific defaults not obvious from the above:
    * density_label = r'$p(' + paramNames[0] + '|\mathrm{data})$'
    * label_kwargs = {'size':16}
    * tick_kwargs = {'labelsize':14, 'length':5}
    * ci1D_kwargs = {'plot_mode':False, 'plot_levels':False, 'plot_ci':False}
    """
    n = len(tri) # how many parameters?
    # defaults
    # plot ranges for each parameter; default to range of corresponding histogram
    if ranges is None:
        ranges = [None]*n
    for i in range(n):
        if ranges[i] is None:
            ranges[i] = [tri[i][i][0]['x'].min(), tri[i][i][0]['x'].max()]
    # y axis maximum of 1D plot at the head of each column
    if Pmax is None:
        Pmax = [None]*n
    # label for the upper-left y axis
    if density_label is None and paramNames is not None:
        density_label = r'$p(' + paramNames[0] + '|\mathrm{data})$'
    if label_kwargs is None:
        label_kwargs = {'size':16}
    if tick_kwargs is None:
        tick_kwargs = {'labelsize':14, 'length':5}
    if ci1D_kwargs is None:
        ci1D_kwargs = {'plot_mode':False, 'plot_levels':False, 'plot_ci':False, 'fill_ci':fill1D}
    if ci2D_kwargs is None:
        ci2D_kwargs = {'fill':fill2D}
    # colors for 1D pdfs, contour lines, contour fill
    try:
        d = ci1D_kwargs['pdf_kwargs']
    except KeyError:
        d = {}
        ci1D_kwargs['pdf_kwargs'] = d
    try:
        d['color']
    except KeyError:
        d['color'] = linecolor1D
    try:
        d = ci2D_kwargs['Line2D_kwargs']
    except KeyError:
        d = {}
        ci2D_kwargs['Line2D_kwargs'] = d
    try:
        d['color']
    except KeyError:
        d['color'] = linecolor2D
    try:
        d['linestyle']
    except KeyError:
        d['linestyle'] = linestyle2D
    try:
        ci1D_kwargs['fill_colors']
        add_fill1D_colors = False
    except KeyError:
        add_fill1D_colors = True
    try:
        ci2D_kwargs['fill_kwargs']['color']
        add_fill2D_colors = False
    except KeyError:
        add_fill2D_colors = True
    try:
        ci2D_kwargs['fill_kwargs']
    except KeyError:
        ci2D_kwargs['fill_kwargs'] = {}
    # do things
    fig = None
    makefig = False
    if axes is None:
        makefig = True
        fig = plt.figure(figsize=(width, width))
        axes = []
    for i in range(n):
        if makefig: axes.append([])
        for j in range(i):
            if makefig:
                ax = fig.add_subplot(n, n, (i*n)+j+1)
            else:
                ax = axes[i][j]
            if not skip2D:
                for k in range(len(tri[i][j]['contours'])-1, -1, -1):
                    if add_fill2D_colors:
                        if fillcolors2D is None:
                            ci2D_kwargs['fill_kwargs']['color'] = str(tri[i][j]['level'][k])
                        else:
                            ci2D_kwargs['fill_kwargs']['color'] = fillcolors2D[k]
                    ci2D_plot(tri[i][j]['contours'][k], ax, **ci2D_kwargs)
            if makefig:
                # Axis limits
                ax.set_xlim(*ranges[j])
                ax.set_ylim(*ranges[i])
                # Tick labels without offset and scientific notation
                ax.get_xaxis().get_major_formatter().set_useOffset(False)
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_useOffset(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                # x-labels at bottom of plot only
                if i == n-1:
                    if paramNames is not None:
                        ax.set_xlabel(paramNames[j], **label_kwargs)
                else:
                    ax.get_xaxis().set_ticklabels([])
                # y-labels for left-most panels only
                if j == 0:
                    if paramNames is not None:
                        ax.set_ylabel(paramNames[i], **label_kwargs)
                else:
                    ax.get_yaxis().set_ticklabels([])
                ax.tick_params(axis='both', **tick_kwargs)
                # Panel layout
                ax.grid(grid)
                axes[i].append(ax)
        if makefig:
            ax = fig.add_subplot(n, n, (i*n)+i+1)
        else:
            ax = axes[i][i]
        if not skip1D:
            if add_fill1D_colors:
                if fillcolors1D is None:
                    ci1D_kwargs['fill_colors'] = [ str(lev) for lev in tri[i][i][1]['level'] ]
                else:
                    ci1D_kwargs['fill_colors'] = fillcolors1D
            ci1D_plot(*tri[i][i], ax, **ci1D_kwargs)
        if makefig:
            # Axis limits
            ax.set_xlim(*ranges[i])
            # Panel layout
            ax.grid(grid)
            # No ticks or labels on y-axes, lower limit 0
            if i > 0:
                ax.yaxis.set_ticks([])
            if Pmin is None:
                ax.set_ylim(bottom=tri[i][i][0]['density'].min())
            else:
                ax.set_ylim(bottom=Pmin)
            if Pmax[i] is not None:
                ax.set_ylim(top=Pmax[i])
            # x-label for bottom-right panel only
            if i == n-1:
                if paramNames is not None:
                    ax.set_xlabel(paramNames[i], **label_kwargs)
            else:
                ax.get_xaxis().set_ticklabels([])
            # y axis label for upper-left
            if i == 0 and density_label is not None:
                ax.set_ylabel(density_label, **label_kwargs)
            ax.tick_params(axis='both', **tick_kwargs)
            axes[i].append(ax)        
    if makefig:
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(wspace=0)
    return fig, axes



def cov_ellipse(cov, center=np.zeros(2), level=0.683, plot=None, fmt='-', npts=100, **kwargs):
    """
    Useful function for plotting the error ellipse associated with a
    2-d Gaussian covariance matrix. Follows the post below, only
    with the size done correctly.
    https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    Returns (xarray, yarray) outline of the ellipse.
    If plot is set to a matplotlib thingy, additionally plots the ellipse.
    """
    theta = np.arange(npts) / (npts-1.0) * 2.0*np.pi
    sx = np.sqrt(cov[0,0])
    sy = np.sqrt(cov[1,1])
    rho = cov[0,1] / (sx*sy)
    x = np.cos(theta) * np.sqrt(1.0 + rho)
    y = np.sin(theta) * np.sqrt(1.0 - rho)
    pa = 0.25 * np.pi
    scl = np.sqrt(st.chi2.ppf(level, 2))
    newx = (x*np.cos(pa) - y*np.sin(pa)) * sx*scl + center[0]
    newy = (x*np.sin(pa) + y*np.cos(pa)) * sy*scl + center[1]
    if plot is not None:
        plot.plot(newx, newy, fmt, **kwargs)
    return newx, newy

