"""Plot functions for the Model class

This contains several plot functions to present results of the models"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

plt.rc('font', size=14)

df = pd.read_csv("scripts/data/data.csv", index_col="t")
df["P"] = df.P_ccc / df.ccc_percent / 10
df["V"] = df.X / df.X_conc
df["X"] = df.X - df.P


def calc_all(model):
    """calculates the required values for the ploting functions"""
    results = model.results.copy()
    results["cX"] = results.X / results.V
    results["cP"] = results.P / results.V
    results["P_per_X"] = results.P / results.X
    results["dP_per_dX"] = results.qP / (results.qG - results.qP - model.qm)
    results["maint"] = model.qm / results.qG
    results["qP_percent"] = results.qP / model.Ypg / results.qG * 100
    results["G_feed"] = results.qG * results.X
    return results


def plot_results(
    *args, filename=None, colors=["b", "g", "m", "c", "k", "r", "y"], style=["-", "--", '-.', (0, (5, 5))]
):
    """plots a summary of results for multible models
    plot_results(*args, filename=None, colors=["b", "g", "m", "y", "b", "r", "c"], style=["-", "--", '-.', ":"]
    plot_results(model1, model2, ..., label1, label2, colors, style)
    model1, model2 ...: models of the model class
    label1, label2, ...: lables for the models as strings
    colors: list of uses colors
    style: list of used linestyles""" 
    # args includes used model followed by the labels of the models.

    model = args[: len(args) // 2]
    labels = args[len(args) // 2 :]

    fig, [(ax0, ax1, ax2), (ax3, ax4, ax5)] = plt.subplots(2, 3, figsize=(15, 7.5))

    for model, label, c, ls in zip(model, labels, colors, style):
        results = calc_all(model)

        ax0.plot(results.X, label=label, c=c, ls=ls)
        ax1.plot(results.P, c=c, ls=ls)
        ax2.plot(results.P_per_X, c=c, ls=ls)
        ax3.plot(results.mu, c=c, ls=ls)
        ax4.plot(results.G_feed, c=c, ls=ls)
        ax5.plot(results.qP_percent, c=c, ls=ls)

        [
            ax.axvline(x=t, ls="dotted", c=c)
            for t in model.phase_switches()
            for ax in (ax0, ax1, ax2, ax3, ax4, ax5)
        ]

    ax3.set_ylim(0, 0.5)

    [ax.set_xlabel("Time $[h]$") for ax in (ax3, ax4, ax5)]
    ax0.set_ylabel("Biomass $[g \\ DW]$")
    ax1.set_ylabel("Product $[g]$")
    ax2.set_ylabel("Product per Biomass")
    ax3.set_ylabel("Growth rate")
    ax4.set_ylabel("Fed glucose $[g \\ h^{-1}]]$")
    ax5.set_ylabel("Glucose used for product $[\%]$")

    # [axs[i, j].legend() for i in range(3) for j in range(4)]

    fig.tight_layout()

    fig.subplots_adjust(bottom=0.15)
    fig.legend(loc="lower center", ncol=len(args) // 2)

    [
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 2))
        for ax in (ax0, ax1, ax2, ax3, ax4, ax5)
    ]

    if filename is not None:
        plt.savefig(filename, facecolor="white", bbox_inches="tight")

    plt.show()


def plot_results_all(
    *args, filename=None, colors=["b", "g", "m", "c", "b", "r", "y"], style=["-", "--", '-.', ":"]
):
    """plots a detailed summary of results for multible models
    plot_results(*args, filename=None, colors=["b", "g", "m", "y", "b", "r", "c"], style=["-", "--", '-.', ":"]
    plot_results(model1, model2, ..., label1, label2, colors, style)
    model1, model2 ...: models of the model class
    label1, label2, ...: lables for the models as strings
    colors: list of uses colors
    style: list of used linestyles""" 

    model = args[: len(args) // 2]
    labels = args[len(args) // 2 :]

    fig, axs = plt.subplots(3, 4, figsize=(15, 7.5))

    for model, label, c, ls in zip(model, labels, colors, style):
        results = calc_all(model)

        axs[0, 0].plot(results.X, label=label, c=c, ls=ls)
        axs[0, 1].plot(results.P, label=label, c=c, ls=ls)
        axs[0, 2].plot(results.V, label=label, c=c, ls=ls)
        axs[1, 0].plot(results.cX, label=label, c=c, ls=ls)
        axs[1, 1].plot(results.cP, label=label, c=c, ls=ls)
        axs[1, 2].plot(results.P_per_X, label=label, c=c, ls=ls)
        axs[0, 3].plot(results.qP, label=label, c=c, ls=ls)
        axs[1, 3].plot(results.qG, label=label, c=c, ls=ls)
        axs[2, 0].plot(results.maint, label=label, c=c, ls=ls)
        axs[2, 1].plot(results.mu, label=label, c=c, ls=ls)
        axs[2, 2].plot(results.G_feed, label=label, c=c, ls=ls)
        axs[2, 3].plot(results.qP_percent, label=label, c=c, ls=ls)

        [
            axs[i, j].axvline(x=t, ls="dotted", c=c)
            for t in model.phase_switches()
            for i in range(3)
            for j in range(4)
        ]

    axs[0, 3].set_title("qP")
    axs[1, 3].set_title("qG")
    axs[1, 3].set_ylim(0, 1)
    axs[2, 0].set_title("growth rate")
    axs[2, 0].set_ylim(0, 0.5)
    axs[2, 1].set_title("maintanance [%]")
    axs[0, 0].set_title("biomass [g DW]")
    axs[0, 1].set_title("product [g DW]")
    axs[0, 2].set_title("volume [L]")
    axs[1, 0].set_title("biomass concentration [g DW/L]")
    axs[1, 1].set_title("product_concentration [g DW/L]")
    axs[1, 2].set_title("product per biomass [g DW/g DW]")
    axs[2, 2].set_title("feeded glucose")
    axs[2, 3].set_title("qP [%]")

    [axs[i, j].set_xlabel("Time [h]") for i in range(2, 3) for j in range(4)]

    # [axs[i, j].legend() for i in range(3) for j in range(4)]
    axs[0, 0].legend()

    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, facecolor="white", bbox_inches="tight")

    plt.show()


def plot_with_data(
    *args,
    filename=None,
    colors=["r", "c", "m", "y", "b", "b", "g"],
    style=["-", "--", "-.", ":"],
    show_data="lin"
):
    """plots the results of the multiple models and the data
    plot_with_data(*args, filename=None, colors=["r", "c", "m", "y", "b", "b", "g"],
    style=["-", "--", "-.", ":"], show_data="lin"
    model1, model2 ...: models of the model class
    label1, label2, ...: lables for the models as strings
    colors: list of uses colors
    style: list of used linestyles
    show_data: "lin", "exp" or "all" to show data with linear feed, with exponential feed or all data"""
    models = args[: len(args) // 2]
    labels = args[len(args) // 2 :]
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(15, 5))

    if show_data == "lin":
        df_plot = df[df.feed == "lin"]
    elif show_data == "exp":
        df_plot = df[df.feed == "exp"]
    else:
        df_plot = df

    for model, label, c, s in zip(models, labels, colors, style):
        results = calc_all(model)
        ax0.plot(results.X, label=label, c=c, linestyle=s)
        ax1.plot(results.P, c=c, linestyle=s)
        ax2.plot(results.P_per_X, c=c, linestyle=s)

        [
            ax.axvline(x=t, ls="dotted", c=c)
            for t in model.phase_switches()
            for ax in (ax0, ax1, ax2)
        ]

    ax0.set_ylabel("Biomass $[g \\ DW]$")
    ax1.set_ylabel("Product $[g]$")
    ax2.set_ylabel("Product per Biomass")

    if show_data in ['lin', 'exp']:
        seaborn.scatterplot(
            x=df_plot.index,
            y=df_plot.X,
            ax=ax0,
            style=df_plot.starvation,
        )
        seaborn.scatterplot(
            x=df_plot.index,
            y=df_plot.P,
            ax=ax1,
            style=df_plot.starvation,
            legend=False,
        )

        seaborn.scatterplot(
            x=df_plot.index,
            y=df_plot.P / df_plot.X,
            ax=ax2,
            style=df_plot.starvation,
            legend=False,
        )
    else:
        seaborn.scatterplot(
            x=df_plot.index,
            y=df_plot.X,
            ax=ax0,
            hue=df_plot.feed,
            style=df_plot.starvation,
        )
        seaborn.scatterplot(
            x=df_plot.index,
            y=df_plot.P,
            ax=ax1,
            hue=df_plot.feed,
            style=df_plot.starvation,
            legend=False,
        )

        seaborn.scatterplot(
            x=df_plot.index,
            y=df_plot.P / df_plot.X,
            ax=ax2,
            hue=df_plot.feed,
            style=df_plot.starvation,
            legend=False,
        )

    [ax.set_xlabel("Time $[h]$") for ax in (ax0, ax1, ax2)]

    fig.tight_layout()
    ax0.get_legend().remove()
    if len(models) <= 4:
        fig.subplots_adjust(bottom=0.22)
    else:
        fig.subplots_adjust(bottom=0.3)
    fig.legend(loc="lower center", ncol=5)

    [
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 2))
        for ax in (ax0, ax1, ax2)
    ]

    if filename is not None:
        plt.savefig(filename, facecolor="white", bbox_inches="tight")

    plt.show()


def energy_usage(*models, model_titels=None, filename=None, percent_only=False):
    """energy_usage(model, model_titels, filename=None, percent_only=False):
    plots the usage of glucose in the model distrubuted between biomass, product, cell maintenance and an overflow process 
    *model: models of the Model class, if percent only=True only a single model can be given
    model_titels: list of model names used in plot title
    filename: if given, saves the plot under filename
    percent_only=False: creates just 1 plot, giving the relative proportion in %
    percent_only=True: creates 2 plots, the second show the absolute values per biomass"""

    no_models = len(models)
    if percent_only:
        if no_models == 1:
            fig, axs = plt.subplots(1, 1, figsize=[7, 5])
        else:
            fig, axs = plt.subplots(no_models // 2, 2, figsize=[15, 5 * no_models // 2])

    if model_titels is None:
        model_titels = [""] * no_models

    for i, (model, title) in enumerate(zip(models, model_titels)):
        df = calc_all(model)
        df["G_P"] = df.qP / model.Ypg
        df["G_X"] = df.mu / model.Yxg
        df["waste"] = df.qG - df.G_X - df.G_P - model.qm
        df["G_m"] = model.qm

        if percent_only:
            if no_models == 1:
                ax = axs
            else:
                ax = axs[i // 2, i % 2]
            
            x = df.index.values
            y0 = df.G_m / df.qG * 100
            y1 = df.G_P / df.qG * 100
            y2 = df.G_X / df.qG * 100
            y3 =df.waste / df.qG * 100
            if i == 0:
                ax.fill_between(x, 0, y0, facecolor='none', edgecolor='b', hatch='//', label='Maintenance')
                ax.fill_between(x, y0, y0+y1, facecolor='none', edgecolor='g', hatch='\\\\', label='Product')
                ax.fill_between(x, y0+y1, y0+y1+y2, facecolor='none', edgecolor='m', hatch='--', label='Biomass')
                ax.fill_between(x, y0+y1+y2, y0+y1+y2+y3, facecolor='none', edgecolor='c', hatch='||', label='Overflow')
            else:
                ax.fill_between(x, 0, y0, facecolor='none', edgecolor='b', hatch='//')
                ax.fill_between(x, y0, y0+y1, facecolor='none', edgecolor='g', hatch='\\\\')
                ax.fill_between(x, y0+y1, y0+y1+y2, facecolor='none', edgecolor='m', hatch='--')
                ax.fill_between(x, y0+y1+y2, y0+y1+y2+y3, facecolor='none', edgecolor='c', hatch='||')
          
            if no_models - i <= 2:                
                ax.set_xlabel("Time [h]")
            if not i % 2:
                ax.set_ylabel("glucose usage [%]")
            ax.set_title(title)
            
        else:
            fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[10, 5])

            x = df.index.values
            y0 = df.G_m
            y1 = df.G_P
            y2 = df.G_X
            y3 =df.waste

            ax1.fill_between(x, 0, y0, facecolor='none', edgecolor='b', hatch='//', label='Maintenance')
            ax1.fill_between(x, y0, y0+y1, facecolor='none', edgecolor='g', hatch='\\\\', label='Product')
            ax1.fill_between(x, y0+y1, y0+y1+y2, facecolor='none', edgecolor='m', hatch='--', label='Biomass')
            ax1.fill_between(x, y0+y1+y2, y0+y1+y2+y3, facecolor='none', edgecolor='c', hatch='||', label='Overflow')

            y0 = df.G_m / df.qG * 100
            y1 = df.G_P / df.qG * 100
            y2 = df.G_X / df.qG * 100
            y3 =df.waste / df.qG * 100

            ax2.fill_between(x, 0, y0, facecolor='none', edgecolor='b', hatch='//')
            ax2.fill_between(x, y0, y0+y1, facecolor='none', edgecolor='g', hatch='\\\\')
            ax2.fill_between(x, y0+y1, y0+y1+y2, facecolor='none', edgecolor='m', hatch='--')
            ax2.fill_between(x, y0+y1+y2, y0+y1+y2+y3, facecolor='none', edgecolor='c', hatch='||')
            
            ax1.set_ylim(0, 0.5)
            ax1.set_xlabel("Time [h]")
            ax1.set_ylabel("Glucose usage $[g \\ (g \\ DW)^{-1}]$")
            # ax1.set_title("Glucose usage per g biomass")

            ax2.set_xlabel("Time [h]")
            ax2.set_ylabel("Glucose usage $[\%]$")
            # ax2.set_title("Glucose usage in %")

    fig.tight_layout()
    if no_models <= 2:
        fig.subplots_adjust(bottom=0.22)
    else:
        fig.subplots_adjust(bottom=0.1)
    fig.legend(loc="lower center", ncol=4)

    if filename is not None:
        plt.savefig(filename, facecolor="white", bbox_inches="tight")

    plt.show()
