import os
import pandas as pd
import numpy as np

import autorank

STUDENT_MODEL = "model_275.pth"
L2_MODEL = "model_40"
COS_MODEL = "swiftdream"

ZINC_MODEL = "model_400"

SINGLE_TEACHER_BERT = "honestcapybara"
SINGLE_TEACHER_TDINFO = "neatspaceship"
TWO_TEACHER = "twoteach"

SMALL_KERNEL = "small_kern"
LARGE_KERNEL = "largekern"

TEACHER_LIST = [
    "GraphMVP",
    "GraphLog",
    "GraphCL",
    "ChemBertMTR-5M",
    "ChemBertMTR-10M",
    "ChemBertMTR-77M",
    "DenoisingPretrainingPQCMv4",
    "FRAD_QM9",
    "ThreeDInfomax",
]


def get_all_results(
    MODELS_TO_EVAL,
    path,
    DATASETS,
    renames=[
        (STUDENT_MODEL, "student-2M"),
        (ZINC_MODEL, "student-250k"),
        (L2_MODEL, "L2"),
        (COS_MODEL, "Cosine"),
        (SMALL_KERNEL, "2-layers-kernel"),
        (LARGE_KERNEL, "5-layers-kernel"),
        (SINGLE_TEACHER_BERT, "sgl-bert"),
        (SINGLE_TEACHER_TDINFO, "sgl-tdinfomax"),
        (TWO_TEACHER, "2-teachers"),
    ],
):
    dfs = []
    for model in MODELS_TO_EVAL:
        model_path = os.path.join(path, model)
        for file in os.listdir(model_path):
            if file.endswith(".csv"):
                df = get_result_model(
                    file, model_path, DATASETS, model, renames=renames
                )
                if not df is None:
                    dfs.append(df)
            else:
                model_path = os.path.join(model_path, file)
                for file in os.listdir(model_path):
                    if file.endswith(".csv"):
                        df = get_result_model(
                            file, model_path, DATASETS, model, renames=renames
                        )
                        if not df is None:
                            dfs.append(df)
                continue

    return pd.concat(dfs)


def get_result_model(
    file,
    model_path,
    DATASETS,
    model,
    renames=[],
    rename_teacher=True,
):
    dataset = file.replace(".csv", "").replace("results_", "")
    if dataset in DATASETS:
        df = pd.read_csv(os.path.join(model_path, file), index_col=0)
        for r in renames:
            model = model.replace(r[0], r[1])
        if rename_teacher:
            if model in TEACHER_LIST:
                model = model + "${}^{(t)}$"
        df["embedder"] = model
        df["dataset"] = dataset
        return df


def rename_cyp(x):
    if "CYP" in x:
        if "(s)" in x or "Substrate" in x:
            return "CYP_(s)"
        return "CYP"
    return x


def aggregate_results_with_ci(df_base, merge_cyp=False):
    if merge_cyp:
        df_base["dataset"] = df_base["dataset"].apply(rename_cyp)
    df_m = df_base.groupby(["dataset", "embedder"]).metric_test.mean().reset_index()
    df_m["dataset"] = df_m["dataset"] + " mean"
    df_v = df_base.groupby(["dataset", "embedder"]).metric_test.std().reset_index()
    df_v["dataset"] = df_v["dataset"] + " std"

    df = df_m.pivot_table(
        index="embedder", columns="dataset", values="metric_test"
    ).join(df_v.pivot_table(index="embedder", columns="dataset", values="metric_test"))
    df.dropna(axis=1, inplace=True)
    # drop column and index names
    df.index.name = None
    order = df.mean(axis=1).sort_values(ascending=False).index.tolist()

    df.columns = pd.MultiIndex.from_tuples(
        [
            (
                df_metadata.loc[c.split(" ")[0], "category"],
                df_metadata.loc[c.split(" ")[0], "short_name"]
                + c.split(" ")[1].replace("mean", "").replace("std", " std"),
            )
            for c in df.columns
        ]
    )

    df[(" ", "avg")] = df_m.pivot_table(
        index="embedder", columns="dataset", values="metric_test"
    ).mean(axis=1)
    df[(" ", "avg std")] = df_m.pivot_table(
        index="embedder", columns="dataset", values="metric_test"
    ).std(axis=1)

    df = df.loc[order[::-1], :]
    df.index = df.index.str.replace("_", " ")

    df = df[sorted(df.columns, key=lambda x: x[0])]

    return df, order


def style_df_ci(df, order, multicols=True, rotate="+"):
    order = [n.replace("_", " ") for n in order]

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.round(x, 3))
    # Get max values
    maxs_vals = df.max(axis=0)
    maxs = df == maxs_vals
    # Get second max values
    df2 = df.where(~maxs)
    maxs_vals = df2.max(axis=0)
    maxs2 = (df2 == maxs_vals) & ~maxs

    style = df.copy()
    for col in style.columns:
        if not col[1].endswith("std"):
            style[col] = (
                style[col].apply(lambda x: f"{x:.3f}")
                + "$\pm$ \\tiny "
                + style[(col[0], col[1] + " std")].apply(lambda x: f"{x:.3f}")
            )
    style = style.loc[order]

    style.drop(
        columns=[
            (col[0], col[1] + " std")
            for col in style.columns
            if not col[1].endswith("std")
        ],
        inplace=True,
    )

    for col in style.columns:
        for best in maxs[maxs[col]].index:
            style.loc[best, col] = "\\textbf{\\underline{" + style.loc[best, col] + "}}"
        for best in maxs2[maxs2[col]].index:
            style.loc[best, col] = "\\textbf{" + style.loc[best, col] + "}"

    if rotate == "+":
        col_prefix = "\\rotatebox{90}{\\shortstack{"
    else:
        col_prefix = "\\rotatebox{-90}{\\shortstack{"
    style.columns = [
        col_prefix + col[0] + " \\\\ " + col[1] + "}}" for col in style.columns
    ]

    style = style.style
    col_format = "r|"

    prev_cols = "This is not a column name that will be used"
    for col in style.columns:
        ov_col = col[len(col_prefix) :].split(" \\\\")[0]
        if prev_cols != ov_col:
            col_format += "|"
            prev_cols = ov_col
        col_format += "c"
    col_format += "|"

    latex = style.to_latex(
        column_format=col_format,
        siunitx=True,
    )
    return style, latex


def get_ranked_df(df_base):
    ranked_df = pd.DataFrame({"embedder": df_base.embedder.unique()})
    for dataset in df_base.dataset.unique():
        df_to_rank = df_base[df_base.dataset == dataset]
        df_to_rank = df_to_rank.pivot_table(
            index="id", columns="embedder", values="metric_test"
        )
        results = autorank.autorank(
            df_to_rank,
            alpha=0.05,
            verbose=True,
            force_mode="nonparametric",
        ).rankdf.reset_index()
        results[dataset] = results["meanrank"]
        results = results[["embedder", dataset]]
        ranked_df = ranked_df.merge(results, on="embedder", how="outer")
    order = ranked_df.mean().sort_values().index.tolist()
    return ranked_df


def add_hline(latex, index, hline=r"\midrule"):
    """
    Adds a horizontal `index` lines before the last line of the table

    Args:
        latex: latex table
        index: index of horizontal line insertion (in lines)
    """
    lines = latex.splitlines()
    if index < 0:
        index = len(lines) + index - 1
    else:
        index = index + 1
    lines.insert(index, hline)
    return "\n".join(lines).replace("NaN", "")


def style_df_ranked(df_ranked, order, avg_task=True, highlight2=True, highlight3=True):
    df_ranked.set_index("embedder", inplace=True)
    df_ranked = df_ranked.loc[order, :]
    df_ranked.index = df_ranked.index.str.replace("_", " ")
    for col in df_ranked.columns:
        df_ranked[col] = df_ranked[col].apply(lambda x: np.round(x, 3))
    # Get max values
    df_ranked.columns = pd.MultiIndex.from_tuples(
        [
            (df_metadata.loc[c, "category"], df_metadata.loc[c, "short_name"])
            for c in df_ranked.columns
        ]
    )
    if avg_task:
        df_ranked = df_ranked.mean(level=0, axis=1)
    df_ranked["Avg"] = df_ranked.mean(axis=1)

    # sort by avg

    min_vals = df_ranked.min(axis=0)
    mins = df_ranked == min_vals

    df2 = df_ranked.where(~mins)
    mins_vals = df2.min(axis=0)
    mins2 = (df2 == mins_vals) & ~mins

    df3 = df2.where(~mins2)
    mins_vals = df3.min(axis=0)
    mins3 = (df3 == mins_vals) & ~mins2

    df_ranked.index.name = None
    style = df_ranked.copy()
    for col in style.columns:
        style[col] = style[col].apply(lambda x: f"{x:.2f}")

    for col in style.columns:
        for best in mins[mins[col]].index:
            style.loc[best, col] = "\\textbf{\\underline{" + style.loc[best, col] + "}}"
        if highlight2:
            for best in mins2[mins2[col]].index:
                style.loc[best, col] = "\\textbf{" + style.loc[best, col] + "}"
        if highlight3:
            for best in mins3[mins3[col]].index:
                style.loc[best, col] = "\\underline{" + style.loc[best, col] + "}"

    cols = style.columns
    col_order = []

    for col in cols:
        sorted_cols = sorted(cols)
    if "Excretion" in cols:
        sorted_cols.remove("Excretion")
        sorted_cols.insert(-1, "Excretion")
    if "HTS" in cols:
        sorted_cols.remove("HTS")
        sorted_cols = sorted_cols + ["HTS"]
    col_order = sorted_cols

    if "Avg" in cols:
        col_order.remove("Avg")
        col_order = col_order + ["Avg"]

    style = style[col_order]

    style = style.style
    col_format = "r"

    for col in style.columns:
        col_format += "|c"
    col_format += "|"

    latex = style.to_latex(
        column_format=col_format,
        multicol_align="|c|",
        siunitx=True,
    )
    return style, latex


df_metadata = pd.read_csv("molDistill/df_metadata.csv").set_index("dataset")
