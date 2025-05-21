import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import font_manager as fm

# /kang/carbon/tsfm_carbon/fonts/helvetica-bold.ttf
font_path = "/kang/carbon/tsfm_carbon/fonts/helvetica-bold.ttf"

# Define font properties
title_size = 36
font_prop_title = fm.FontProperties(fname=font_path, size=title_size)
font_prop_ssim = fm.FontProperties(fname=font_path, size=20)

label_size = 28
ticks_size = 28
legend_size = 24
marker_size = 16
line_width = 9

font_prop_ticks = fm.FontProperties(fname=font_path, size=ticks_size)
font_prop_label = fm.FontProperties(fname=font_path, size=label_size)
font_prop_legend = fm.FontProperties(fname=font_path, size=legend_size)
font_prop_annotate = fm.FontProperties(fname=font_path, size=marker_size)


# Parameters
# Change this to "1D", "7D", "30D", etc. for daily, weekly, monthly

duration = "1D"  # 24h
index_list = [0, 4, 5]
region_list_m = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]

regison_dict = {
    "CISO": "California (CISO)",
    "PJM": "Pennsylvania-Jersey-Maryland (PJM)",
    "ERCO": "Texas (ERCOT)",
    "ISNE": "New England (ISO-NE)",
    "SE": "Sweden (SE)",
    "DE": "Germany (DE)"
}
# region_list_m = ["California (CISO)", "Pennsylvania-Jersey-Maryland (PJM)", "Texas (ERCOT)", "New England (ISO-NE)", "Sweden (SE)", "Germany (DE)"]

region_list = [region_list_m[i] for i in index_list]

colors_m = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
colors = [colors_m[i] for i in index_list]

# colors = ['#4363d8', '#e6194b', '#3cb44b', '#ffe119', 
#               '#f032e6', '#fabebe', '#008080', 'black','#e6beff', '#9a6324', '#fffac8', 
#               '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

linestyles = ["-", "--", "-.", ":", "-", "--"]
markers = ["o", "s", "D", "^", "v", "P"]

# plt.figure(figsize=(12, 6))
plt.figure(figsize=(6.4 * 4, 4.8))

# for region in region_list:
for i, region in enumerate(region_list):

    data_base_dir_region = f"/kang/carbon/CarbonCast_main/data/{region}"
    in_file_name = os.path.join(data_base_dir_region, f'{region}_lifecycle_emissions.csv')

    # Load dataset
    df = pd.read_csv(in_file_name, header=0, parse_dates=['UTC time'], index_col='UTC time')

    # Take only the first year
    first_year = df.loc[df.index < (df.index[0] + pd.DateOffset(years=1))]

    second_year = df.loc[(df.index >= df.index[0] + pd.DateOffset(years=1)) &
                     (df.index < df.index[0] + pd.DateOffset(years=2))]

    year_data = second_year

    # Extract the second column
    values = year_data.iloc[:, 1]

    # # Take only the second column (assuming lifecycle emissions are there)
    # values = df.iloc[:, 1]

    # Resample and average over duration
    averaged = values.resample(duration).mean()

    # Plot
    # plt.plot(averaged.index, averaged.values, label=region)
    plt.plot(
        averaged.index,
        averaged.values,
        label=regison_dict[region],
        linewidth=line_width,
        # linestyle=linestyles[i],
        # marker=markers[i],
        # markersize=5,
        color=colors[i],
        # markerfacecolor='white',
        # markeredgewidth=1.2
    )


# fig_path = os.path.join("./figures", f'lifecycle_emissions_{duration}.png')
# plt.title(f"Lifecycle Emissions Averaged Over {duration}")
plt.xlabel("Date (aggregating hourly data per day)", fontproperties=font_prop_label)
plt.ylabel("Intensity (gCO2/kWh)", fontproperties=font_prop_label)


plt.xticks(fontproperties=font_prop_ticks)
plt.ylim(0, 550)             
plt.yticks(ticks=range(0, 501, 100), fontproperties=font_prop_ticks)

# plt.legend(ncol=4, prop=font_prop_legend)

plt.legend(
    ncol=6,
    prop=font_prop_legend,
    loc='upper left',
    # bbox_to_anchor=(0.5, 1.5),  # position below the plot
    frameon=True,
    fancybox=True,
    framealpha=0.8
)

# plt.legend(prop=helvetica_font)

plt.grid(True)
plt.tight_layout()

filename = os.path.join("./figures", f'lifecycle_emissions_{duration}_year2.pdf')

with PdfPages(filename) as pdf:
    plt.savefig(pdf, format='pdf', dpi=2000)  # Save with 2000 DPI

plt.close()
# plt.show()

