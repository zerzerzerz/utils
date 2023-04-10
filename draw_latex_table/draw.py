import pandas as pd

with open("start.tex") as f:
    start = f.read()

with open("end.tex") as f:
    end = f.read() 

with open("ans.tex", "w") as f:
    f.write(start)

    df = pd.read_csv("data.csv")

    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    modes = sorted(df["mode"].unique())
    metrics = ['mse', 'psnr']

    f.write("\\multicolumn{2}{c|}{\\multirow{2}{*}{Dataset/Model}}")
    for model in models:
        f.write(" & \\multicolumn{%d}{c}{%s}" % ( len(metrics), model ))
    f.write("\\\\\n")

    f.write("\\multicolumn{2}{c|}{~}")
    for model in models:
        for metric in metrics:
            f.write(" & %s" % metric)
    f.write("\\\\\n")

    for mode in modes:
        f.write("\\midrule\n")
        for i, dataset in enumerate(datasets):
            if i == 0:
                f.write("\\multirow{%d}{*}{%s}" % ( len(datasets), mode ))
            else:
                f.write("~")
            
            f.write(" & %s" % dataset)
            
            for model in models:
                for metric in metrics:
                    v = df.loc[
                        (df["mode"] == mode) &
                        (df["dataset"] == dataset) &
                        (df["model"] == model),
                        metric
                    ].values[0]

                    if metric == "mse":
                        v_best = df.loc[
                            (df["mode"] == mode) &
                            (df["dataset"] == dataset),
                            metric
                        ].min()
                    elif metric == 'psnr':
                        v_best = df.loc[
                            (df["mode"] == mode) &
                            (df["dataset"] == dataset),
                            metric
                        ].max()
                    else:
                        raise NotImplementedError
                    
                    if abs(v - v_best) < 1e-6:
                        f.write(" & \\textbf{ %.3f }" % v)
                    else:
                        f.write(" & %.3f" % v)



            f.write("\\\\\n")



    f.write(end)

