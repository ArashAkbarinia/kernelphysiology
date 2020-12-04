import numpy as np
import os

from matplotlib import pyplot as plt
from scipy import stats

mStyles = [
    "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P",
    "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11
]


def get_label_colour(key):
    colour = key
    label = key
    if 'lum_' in label:
        lum_val = 0.5 + float(label.split('_')[-1])
        label = 'lum=%.2f' % lum_val
        marker = '-' + mStyles[int(lum_val * 10)]
        lum_val /= 2
        colour = (lum_val, lum_val, lum_val)
        kwargs = {'color': colour}
    elif colour in ['lum', 'instance1', 'lum_+0.00']:
        colour = 'gray'
        marker = '-x'
        kwargs = {'color': colour}
    elif colour in ['instance2', 'lum_+0.25']:
        colour = 'blue'
        marker = '-+'
        kwargs = {'color': colour}
    elif colour in ['instance3', 'lum_-0.25']:
        colour = 'green'
        marker = '-o'
        kwargs = {'color': colour, 'markerfacecolor': 'white'}
    elif colour in ['instance4']:
        colour = 'orange'
        marker = '-*'
        kwargs = {'color': colour, 'markerfacecolor': 'white'}
    elif colour == 'rg':
        colour = 'green'
        marker = '-1'
        label = 'rg   '
        kwargs = {'color': colour, 'markerfacecolor': 'white',
                  'markeredgecolor': 'r'}
    elif colour == 'yb':
        colour = 'blue'
        marker = '-2'
        label = 'yb   '
        kwargs = {'color': colour, 'markerfacecolor': 'white',
                  'markeredgecolor': 'y'}
    return label, marker, kwargs


def interpolate_all_sfs(xvals, yvals, target_size, max_val):
    base_sf = ((target_size / 2) / np.pi)
    new_xs = [
        base_sf / e for e in np.arange(1, max_val + 0.5, 0.5)
    ]
    new_ys = np.interp(new_xs, xvals, yvals)
    return new_xs, new_ys


def human_csf(f):
    return 2.6 * (0.0192 + 0.114 * f) * np.exp(-(0.114 * f) ** 1.1)


def plot_sensitivity(all_summaries, param_name, out_file=None, target_size=None,
                     nrows=5, figsize=(22, 4), log_axis=False, till64=False,
                     normalise=True):
    if log_axis:
        till64 = True

    for key_exp, exp_results in all_summaries.items():
        fig = plt.figure(figsize=figsize)
        first_axes = True
        axes = []
        avg_plots = []
        for key_chn, all_results in exp_results.items():
            num_tests = len(all_results)
            current_avg = []
            for i in range(num_tests):
                plot_human = True
                cur_sen = all_results[i][0]['sensitivities']
                # getting the x and y values
                if type(param_name) is list:
                    org_yvals = cur_sen[param_name[0]][param_name[1]]
                else:
                    org_yvals = cur_sen[param_name]
                org_xvals = all_results[i][0]['unique_params']['wave']
                # convert to array
                org_xvals = np.array(org_xvals)
                org_yvals = np.array(org_yvals)

                if first_axes:
                    ax = fig.add_subplot(1, nrows, i + 1)
                    parts = all_results[i][1].split('_')
                    if parts[-1] in ['trichromat', 'monochromat']:
                        tmp = parts[-1]
                        parts[-1] = parts[-2]
                        parts[-2] = tmp
                    elif parts[-1] in ['rg', 'yb']:
                        tmp = parts[-1]
                        parts[-1] = parts[-3]
                        parts[-3] = 'dichromat'
                        parts[-2] = tmp
                    title = '%s%d' % (parts[-1][:-1], int(parts[-1][-1]) - 1)
                    title = title.replace('layer', 'area')
                    ax.set_title(title)
                    axes.append(ax)
                else:
                    ax = axes[i]
                current_avg.append(org_yvals)
                if key_chn in ['red', 'green', 'blue']:
                    continue
                label, marker, kwargs = get_label_colour(key_chn)

                base_sf = ((target_size / 2) / np.pi)
                org_freqs = [((1 / e) * base_sf) for e in org_xvals]

                if till64:
                    xsinds = np.argsort(org_freqs)
                    org_freqs = np.array([org_freqs[sind] for sind in xsinds])
                    cut_yvals = np.array([org_yvals[sind] for sind in xsinds])
                    cut_xvals = np.array([org_xvals[sind] for sind in xsinds])

                    ind64 = np.where(org_freqs > 64)
                    if len(ind64[0]) > 0:
                        ind64 = ind64[0][0]
                    else:
                        ind64 = -1
                    org_freqs = org_freqs[:ind64]
                    cut_yvals = cut_yvals[:ind64]
                    cut_xvals = cut_xvals[:ind64]

                    last_freq = org_freqs[-1]
                else:
                    cut_xvals = org_xvals
                    cut_yvals = org_yvals
                    last_freq = org_freqs[0]

                if normalise:
                    cut_yvals /= cut_yvals.max()

                # first plot the human CSF
                if 'lum' in key_chn and plot_human:
                    plot_human = False
                    hcsf = np.array([human_csf(f) for f in org_freqs])
                    hcsf /= hcsf.max()
                    hcsf *= np.max(cut_yvals)
                    ax.plot(org_freqs, hcsf, '--', color='black', label='human')

                # interpolating to all points
                int_xvals, int_yvals = interpolate_all_sfs(
                    org_xvals, org_yvals, target_size, last_freq
                )
                int_freqs = [((1 / e) * base_sf) for e in int_xvals]

                # compute the correlation
                hcsf = np.array([human_csf(f) for f in int_freqs])
                hcsf /= hcsf.max()
                int_yvals /= int_yvals.max()
                p_corr, r_corr = stats.pearsonr(int_yvals, hcsf)
                euc_dis = np.linalg.norm(hcsf - int_yvals)
                ax.plot(
                    org_freqs, cut_yvals, marker,
                    label='%s [r=%.2f | d=%.2f]' % (label, p_corr, euc_dis),
                    **kwargs
                )

                ax.set_xlabel('Spatial Frequency (Cycle/Image)')
                ax.set_ylabel('Sensitivity (1/Contrast)')
                if log_axis:
                    ax.set_xscale('log')
                ax.legend()
            avg_plots.append(np.array(current_avg).mean(axis=0))
            first_axes = False

        if out_file is not None:
            fig.tight_layout()
            fig.savefig(
                os.path.join(out_file, '%s_sensitivity.png' % (key_exp))
            )


def compute_corrs(all_summaries, param_name, target_size=None, animal='human',
                  till64=False):
    corr_networks = dict()
    euc_networks = dict()
    for key_exp, exp_results in all_summaries.items():
        corr_networks[key_exp] = dict()
        euc_networks[key_exp] = dict()
        for key_chn, all_results in exp_results.items():
            num_tests = len(all_results)
            chn_corrs = []
            chn_eucs = []
            for i in range(num_tests):
                cur_sen = all_results[i][0]['sensitivities']
                # getting the x and y values
                if type(param_name) is list:
                    org_yvals = cur_sen[param_name[0]][param_name[1]]
                else:
                    org_yvals = cur_sen[param_name]
                org_xvals = all_results[i][0]['unique_params']['wave']
                # convert to array
                org_xvals = np.array(org_xvals)
                org_yvals = np.array(org_yvals)

                base_sf = ((target_size / 2) / np.pi)
                org_freqs = [((1 / e) * base_sf) for e in org_xvals]

                if till64:
                    xsinds = np.argsort(org_freqs)
                    org_freqs = np.array([org_freqs[sind] for sind in xsinds])

                    ind64 = np.where(org_freqs > 64)
                    if len(ind64[0]) > 0:
                        ind64 = ind64[0][0]
                    else:
                        ind64 = -1
                    org_freqs = org_freqs[:ind64]
                    last_freq = org_freqs[-1]
                else:
                    last_freq = org_freqs[0]

                # interpolating to all points
                int_xvals, int_yvals = interpolate_all_sfs(
                    org_xvals, org_yvals, target_size, last_freq
                )
                int_freqs = [((1 / e) * base_sf) for e in int_xvals]

                # compute the correlation
                hcsf = np.array([human_csf(f) for f in int_freqs])
                hcsf /= hcsf.max()
                int_yvals /= int_yvals.max()
                p_corr, r_corr = stats.pearsonr(int_yvals, hcsf)
                euc_dis = np.linalg.norm(hcsf - int_yvals)

                chn_corrs.append(p_corr)
                chn_eucs.append(euc_dis)
            corr_networks[key_exp][key_chn] = chn_corrs
            euc_networks[key_exp][key_chn] = chn_eucs
    return corr_networks, euc_networks


def group_compute(all_summaries, param_name, target_size=None, animal='human',
                  till64=False):
    freq_networks = dict()
    yval_networks = dict()
    hcsf_networks = dict()
    for key_exp, exp_results in all_summaries.items():
        freq_networks[key_exp] = dict()
        yval_networks[key_exp] = dict()
        hcsf_networks[key_exp] = dict()
        for key_chn, all_results in exp_results.items():
            num_tests = len(all_results)
            all_freqs = []
            all_yvals = []
            all_hcsfs = []
            for i in range(num_tests):
                cur_sen = all_results[i][0]['sensitivities']
                # getting the x and y values
                if type(param_name) is list:
                    org_yvals = cur_sen[param_name[0]][param_name[1]]
                else:
                    org_yvals = cur_sen[param_name]
                org_xvals = all_results[i][0]['unique_params']['wave']
                # convert to array
                org_xvals = np.array(org_xvals)
                org_yvals = np.array(org_yvals)

                base_sf = ((target_size / 2) / np.pi)
                org_freqs = [((1 / e) * base_sf) for e in org_xvals]

                if till64:
                    xsinds = np.argsort(org_freqs)
                    org_freqs = np.array([org_freqs[sind] for sind in xsinds])
                    cut_yvals = np.array([org_yvals[sind] for sind in xsinds])
                    cut_xvals = np.array([org_xvals[sind] for sind in xsinds])

                    ind64 = np.where(org_freqs > 64)
                    if len(ind64[0]) > 0:
                        ind64 = ind64[0][0]
                    else:
                        ind64 = -1
                    org_freqs = org_freqs[:ind64]
                    cut_yvals = cut_yvals[:ind64]
                    cut_xvals = cut_xvals[:ind64]

                    last_freq = org_freqs[-1]
                else:
                    cut_xvals = org_xvals
                    cut_yvals = org_yvals
                    last_freq = org_freqs[0]

                # compute the correlation
                hcsf = np.array([human_csf(f) for f in org_freqs])
                hcsf /= hcsf.max()
                cut_yvals /= cut_yvals.max()

                all_freqs.append(org_freqs)
                all_yvals.append(cut_yvals)
                all_hcsfs.append(hcsf)
            freq_networks[key_exp][key_chn] = all_freqs
            yval_networks[key_exp][key_chn] = all_yvals
            hcsf_networks[key_exp][key_chn] = all_hcsfs
    return freq_networks, yval_networks, hcsf_networks


def group_plot(results, out_file=None, nrows=5, figsize=(22, 4),
               log_axis=False,):
    fig = plt.figure(figsize=figsize)
    freq_networks, yval_networks, hcsf_networks = results
    for i in range(freq_networks.shape[0]):
        ax = fig.add_subplot(1, nrows, i + 1)
        ax.set_title('area%d' % i)

        label, marker, kwargs = get_label_colour('lum')

        xval = freq_networks[i].squeeze()
        yval = yval_networks[i].squeeze()
        hcsf = hcsf_networks[i].squeeze()

        xsinds = np.argsort(xval)
        xval = np.array([xval[sind] for sind in xsinds])
        yval = np.array([yval[sind] for sind in xsinds])
        hcsf = np.array([hcsf[sind] for sind in xsinds])

        if log_axis:
            yval /= yval.max()

        # interpolating to all points
        new_xs = np.array(np.arange(1, xval.max() + 0.5, 0.5))
        new_ys = np.interp(new_xs, xval, yval)

        # compute the correlation
        hcsf_corr = np.array([human_csf(f) for f in new_xs])
        hcsf_corr /= hcsf_corr.max()
        new_ys /= new_ys.max()
        p_corr, r_corr = stats.pearsonr(new_ys, hcsf_corr)
        euc_dis = np.linalg.norm(hcsf_corr - new_ys)

        ax.plot(xval, hcsf, '--', color='black', label='human')
        ax.plot(
            xval, yval, marker,
            label='%s [r=%.2f | d=%.2f]' % (label, p_corr, euc_dis),
            **kwargs
        )

        ax.set_xlabel('Spatial Frequency (Cycle/Image)')
        ax.set_ylabel('Sensitivity (1/Contrast)')
        if log_axis:
            ax.set_xscale('log')
        ax.legend()

    if out_file is not None:
        fig.tight_layout()
        fig.savefig(out_file + '_sensitivity.png')
