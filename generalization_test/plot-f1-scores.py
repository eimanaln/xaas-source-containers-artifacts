import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

def normalize_key(key):
    return key.replace('-', '_').replace(' ', '_').lower()

def normalize_flag(flag):
    if not flag:
        return None
    return flag.strip()

def extract_relevant_fields(json_data):
    flattened = {}
    if "gpu_build" in json_data:
        flattened["gpu_build.value"] = json_data["gpu_build"].get("value")
        flattened["gpu_build.build_flag"] = normalize_flag(json_data["gpu_build"].get("build_flag"))
    if "gpu_backends" in json_data:
        for backend, info in json_data["gpu_backends"].items():
            b_key = normalize_key(backend)
            flattened[f"gpu_backends.{b_key}.used_as_default"] = info.get("used_as_default")
            flattened[f"gpu_backends.{b_key}.build_flag"] = normalize_flag(info.get("build_flag"))

    if "parallel_programming_libraries" in json_data:
        for lib, info in json_data["parallel_programming_libraries"].items():
            p_key = normalize_key(lib)
            flattened[f"parallel_programming_libraries.{p_key}.used_as_default"] = info.get("used_as_default")
            flattened[f"parallel_programming_libraries.{p_key}.build_flag"] = normalize_flag(info.get("build_flag"))
            
    if "linear_algebra_libraries" in json_data:
        for lib, info in json_data["linear_algebra_libraries"].items():
            l_key = normalize_key(lib)
            flattened[f"linear_algebra_libraries.{l_key}.used_as_default"] = info.get("used_as_default")
    if "FFT_libraries" in json_data:
        for lib, info in json_data["FFT_libraries"].items():
            f_key = normalize_key(lib)
            flattened[f"FFT_libraries.{f_key}.built-in"] = info.get("built-in")
            flattened[f"FFT_libraries.{f_key}.used_as_default"] = info.get("used_as_default")
            flattened[f"FFT_libraries.{f_key}.build_flag"] = normalize_flag(info.get("build_flag"))
    if "simd_vectorization" in json_data:
        for simd, info in json_data["simd_vectorization"].items():
            s_key = normalize_key(simd)
            flattened[f"simd_vectorization.{s_key}.build_flag"] = normalize_flag(info.get("build_flag"))
            flattened[f"simd_vectorization.{s_key}.default"] = info.get("default")
    if "build_system" in json_data:
        flattened["build_system.type"] = normalize_key(json_data["build_system"].get("type"))
        flattened["build_system.minimum_version"] = json_data["build_system"].get("minimum_version")
    if "internal_build" in json_data:
        flattened["internal_build.library_name"] = normalize_key(json_data["internal_build"].get("library_name"))
        flattened["internal_build.build_flag"] = normalize_flag(json_data["internal_build"].get("build_flag"))
    return flattened

def compare_dicts(gt, pred):
    tp = fp = fn = 0
    all_keys = set(gt.keys()).union(pred.keys())
    for key in all_keys:
        gt_val = gt.get(key)
        pred_val = pred.get(key)
        if gt_val is not None and pred_val is not None:
            if gt_val == pred_val:
                tp += 1
            else:
                fp += 1
        elif pred_val is not None:
            fp += 1
        elif gt_val is not None:
            fn += 1
    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def evaluate_applications(base_dir, gt_dir):
    apps = {
        "lulesh": "lulesh-specializations",
        "lammps": "lammps-specializations"
    }

    long_metrics = []

    for app_name, app_dir in apps.items():
        gt_path = os.path.join(gt_dir, f"{app_name}.json")
        with open(gt_path, 'r') as f:
            gt_flat = extract_relevant_fields(json.load(f))

        app_path = os.path.join(base_dir, app_dir)
        if not os.path.isdir(app_path):
            print(f"Directory not found: {app_path}")
            continue

        report_dir = os.path.join(app_path, "reports")
        os.makedirs(report_dir, exist_ok=True)

        for pred_file in sorted(os.listdir(app_path)):
            if not pred_file.endswith(".json"):
                continue
            try:
                with open(os.path.join(app_path, pred_file), 'r') as pf:
                    pred_json = json.load(pf)
                    pred_flat = extract_relevant_fields(pred_json)
                    tp, fp, fn = compare_dicts(gt_flat, pred_flat)
                    precision, recall, f1 = compute_metrics(tp, fp, fn)

                    # ✏️ Write report for this prediction
                    report_path = os.path.join(report_dir, f"{os.path.splitext(pred_file)[0]}_report.txt")
                    with open(report_path, 'w') as rf:
                        rf.write(f"Application: {app_name}\n")
                        rf.write(f"Prediction File: {pred_file}\n")
                        rf.write(f"True Positives (TP): {tp}\n")
                        rf.write(f"False Positives (FP): {fp}\n")
                        rf.write(f"False Negatives (FN): {fn}\n")
                        rf.write(f"Precision: {precision:.4f}\n")
                        rf.write(f"Recall: {recall:.4f}\n")
                        rf.write(f"F1 Score: {f1:.4f}\n")

                    long_metrics.extend([
                        {"Application": app_name, "Metric": "F1 Score", "Value": f1},
                        {"Application": app_name, "Metric": "Precision", "Value": precision},
                        {"Application": app_name, "Metric": "Recall", "Value": recall}
                    ])
            except Exception as e:
                print(f"Error processing {pred_file}: {e}")

    return pd.DataFrame(long_metrics)

def plot_metric_boxplots(df):
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    palette = {"lulesh": "#1C64E3", "lammps": "#F45B17"}

    g = sns.catplot(
        data=df,
        x="Application",
        y="Value",
        hue="Application",
        col="Metric",
        col_order=["F1 Score", "Precision", "Recall"],
        palette=palette,
        kind="box",
        height=4.5,
        aspect=0.6,
        legend=True,
        legend_out=True,
    )

    g.set(ylim=(0, 1.1))
    for ax in g.axes.flat:
        title = ax.get_title().split('=')[-1].strip()
        ax.set_title("")
        ax.add_patch(patches.Rectangle((0, 1.02), 1, 0.08, transform=ax.transAxes, color="#AFAEAE", clip_on=False))
        ax.text(0.5, 1.06, title, transform=ax.transAxes, ha="center", va="center", color="white", fontweight="bold", fontsize=12)
        ax.set_ylabel("Score", fontweight='bold', labelpad=15)
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.tick_params(axis='x', colors='grey')

    g.fig.suptitle("Distribution of F1 Score, Precision, and Recall per Application", fontweight='bold', x=0.42)
    g.fig.subplots_adjust(top=0.85, wspace=0.029)
    g._legend.set_title("Application", prop={'weight': 'bold'})

    plt.savefig("Applications-Metrics-Distribution.pdf", format='pdf', bbox_inches='tight')

def print_metric_statistics(df):
    print("\n📊 Metric Summary (Mean, Median, Min, Max per Application/Metric):\n")
    grouped = df.groupby(["Application", "Metric"])["Value"]
    summary = grouped.agg(['mean', 'median', 'min', 'max']).round(4)
    print(summary.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate specialization points for multiple HPC applications.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing application folders.")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Directory containing ground truth JSON files.")
    args = parser.parse_args()

    metrics_df = evaluate_applications(args.base_dir, args.ground_truth_dir)
    print_metric_statistics(metrics_df)
    plot_metric_boxplots(metrics_df)