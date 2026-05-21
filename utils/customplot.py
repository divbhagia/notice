import numpy as np
import matplotlib.pyplot as plt
from utils.config import FONT, Colors, CRIT

##########################################################
# Function to add line breaks to long labels
##########################################################


def add_lines_labs(labels):
    newlabs = labels.copy()
    for i, lab in enumerate(newlabs):
        if len(lab) > 20:
            midpoint = len(lab) // 2
            left_index = lab.rfind(" ", 0, midpoint)
            right_index = lab.find(" ", midpoint)
            if left_index > 0 and lab[left_index - 1] == "&":
                left_index += 1
            if left_index == -1 and right_index == -1:
                split_index = midpoint
            elif left_index == -1:
                split_index = right_index
            elif right_index == -1:
                split_index = left_index
            else:
                split_index = (
                    left_index
                    if (midpoint - left_index) <= (right_index - midpoint)
                    else right_index
                )
            newlabs[i] = lab[:split_index] + "\n" + lab[split_index:].lstrip()
    return newlabs


##########################################################
# Font and other asthestics for plots
##########################################################


def set_plot_aes():
    from utils.config import FONT
    import matplotlib.font_manager as font_manager

    fonts = [font.name for font in font_manager.fontManager.ttflist]
    if FONT["font"] in fonts:
        plt.rcParams[f"font.{FONT['family']}"] = FONT["font"]
        plt.rcParams.update({"font.size": FONT["size"], "font.family": FONT["family"]})
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    return FONT


##########################################################
# Custom Plot
##########################################################


def custom_plot(
    series,
    se=None,
    xlab="",
    ylab="",
    title="",
    legendlabs=None,
    xticklabs=None,
    figsize=[2.9, 2.75],
    ydist=0.1,
    ylims=None,
    crit=CRIT,
    colors=None,
    linestyles=None,
    linewidths=None,
    legendpos="best",
    savepath=None,
    subplot=False,
    legend=True,
    ax=None,
):

    # Initialize
    # matplotlib.use('PDF')
    series = [np.array(s) for s in series]
    num_series = len(series)

    # Check if se is provided
    if se is None:
        se = [None for s in series]
    assert len(se) == num_series, "Length of standard errors must match series"

    # Fonts for elements
    set_plot_aes()
    font_base = {"family": FONT["family"], "size": FONT["size"]}
    font_title = font_base.copy()
    font_axis_labs = font_base.copy()
    font_legend = font_base.copy()
    font_axis_ticks = {"family": FONT["family"], "size": FONT["size"] - 1}

    # Custom font changes
    font_title["size"] = 10
    font_title["weight"] = "medium"

    # Default colors & linestyles
    red = Colors.RED
    black = Colors.BLACK
    blue = Colors.BLUE
    green = Colors.GREEN
    colors = [red, black, blue, green] if colors is None else colors
    linestyles = ["-", "--", ":", "-."] if linestyles is None else linestyles
    if linewidths is None:
        linewidths = len(linestyles) * [1.5]
        for i in range(len(linestyles)):
            if linestyles[i] == ":":
                linewidths[i] = 2

    # Default xtick labels
    if xticklabs is None:
        xticklabs = [f"{t}" for t in range(len(series[0]))]

    # Default legend labels
    if legendlabs is None:
        legendlabs = [f"Series {j+1}" for j in range(num_series)]

    # Y-axis limits
    if ylims is None:
        lb, ub = np.zeros(num_series), np.zeros(num_series)
        for j in range(num_series):
            se_j = np.zeros_like(series[j]) if se[j] is None else np.array(se[j])
            lb[j] = np.min(series[j] - crit * se_j)
            ub[j] = np.max(series[j] + crit * se_j)
        min_, max_ = np.min(lb), np.max(ub)
        lb = min_ - 0.1 * (max_ - min_)
        ub = max_ + 0.1 * (max_ - min_)
    else:
        lb, ub = ylims
    yticks = np.arange(np.round(lb * 10) / 10, ub, ydist)

    # Plot figure
    if subplot is False and ax is None:
        plt.figure(figsize=figsize)
    for j in range(num_series):
        plt.plot(
            series[j],
            label=legendlabs[j],
            color=colors[j],
            linestyle=linestyles[j],
            linewidth=linewidths[j],
        )
        if se[j] is not None:
            plt.errorbar(
                range(len(series[j])),
                series[j],
                yerr=crit * se[j],
                color=colors[j],
                capsize=2.5,
                alpha=1,
                fmt="o",
                markersize=0,
            )
    plt.xlabel(xlab, fontdict=font_axis_labs)
    plt.ylabel(ylab, fontdict=font_axis_labs)
    plt.yticks(yticks)
    plt.ylim(lb, ub)
    plt.xlim(-0.25, len(series[0]) - 1 + 0.25)
    plt.xticks(range(len(series[0])), xticklabs)
    plt.title(title, fontdict=font_title)

    # Customize legend
    if legend:
        plt.legend(
            prop=font_legend,
            loc=legendpos,
            borderaxespad=0.5,
            borderpad=0.25,
            labelspacing=0.25,
            handlelength=2.5,
            framealpha=0.5,
        )

    # # Customize font for tick labels
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontsize(font_axis_ticks["size"])
        label.set_fontname(font_axis_ticks["family"])

    # Adjust layout
    plt.tight_layout(pad=1)
    plt.subplots_adjust(left=0.165)

    # Save figure
    if savepath is not None:
        plt.savefig(savepath, dpi=300, format="pdf", transparent=True)


##########################################################
