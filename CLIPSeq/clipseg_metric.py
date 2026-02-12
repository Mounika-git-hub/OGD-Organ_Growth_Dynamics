import os
import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUAL_LABEL_DIR = os.path.join(BASE_DIR, "manual_lables")
PRED_BASE_DIR = os.path.join(BASE_DIR, "outputs")
LABELED_OUTPUT_DIR = os.path.join(BASE_DIR, "labeled_metric_output")
PROMPT_FOLDER = "bright_biological_tissue"

def dice_score(gt, pred):
    intersection = np.sum((gt == 1) & (pred == 1))
    return (2.0 * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)

def iou_score(gt, pred):
    intersection = np.sum((gt == 1) & (pred == 1))
    union = np.sum((gt == 1) | (pred == 1))
    return intersection / (union + 1e-8)

def precision_score(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    return tp / (tp + fp + 1e-8)

def recall_score(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    return tp / (tp + fn + 1e-8)

def accuracy_score(gt, pred):
    return np.sum(gt == pred) / gt.size

#Helper functions
def load_ground_truth(gt_path):
    gt_stack = tiff.imread(gt_path)
    if gt_stack.ndim == 2:
        gt_stack = gt_stack[np.newaxis, ...]
    return (gt_stack > 0).astype(np.uint8)

def find_prediction_folder(sample_name):
    """
    To Find prediction folder ignoring case sensitivity.
    """
    for folder in os.listdir(PRED_BASE_DIR):
        if folder.lower() == sample_name.lower():
            return os.path.join(
                PRED_BASE_DIR,
                folder,
                "clipseg",
                "masks",
                PROMPT_FOLDER)
    return None

def find_slice_file(pred_mask_dir, z):
    """
    To Find slice file ignoring case differences in sample name.
    """
    expected_suffix = f"slice_{z:03d}.npy"
    for file in os.listdir(pred_mask_dir):
        if file.lower().endswith(expected_suffix.lower()):
            return os.path.join(pred_mask_dir, file)
    return None

def evaluate_sample(sample_name):
    print(f"\nEvaluating {sample_name}:")
    gt_path = os.path.join(MANUAL_LABEL_DIR, f"{sample_name}_label.tif")
    if not os.path.exists(gt_path):
        print("Ground truth not found.")
        return None
    pred_mask_dir = find_prediction_folder(sample_name)
    if pred_mask_dir is None or not os.path.exists(pred_mask_dir):
        print("Prediction folder missing.")
        return None
    gt_stack = load_ground_truth(gt_path)
    results = []
    for z in tqdm(range(gt_stack.shape[0])):
        pred_path = find_slice_file(pred_mask_dir, z)
        if pred_path is None:
            continue
        gt = gt_stack[z]
        pred = np.load(pred_path).astype(np.uint8)
        if gt.shape != pred.shape:
            print(f"Shape mismatch in slice {z}")
            continue
        results.append([
            z,
            dice_score(gt, pred),
            iou_score(gt, pred),
            precision_score(gt, pred),
            recall_score(gt, pred),
            accuracy_score(gt, pred)])

    if not results:
        print("No matching slices found.")
        return None
    df = pd.DataFrame(results, columns=["Slice", "Dice", "IoU", "Precision", "Recall", "Accuracy"])
    mean_values = df.mean(numeric_only=True)
    mean_values["Slice"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([mean_values])], ignore_index=True)
    return df

def save_sample_outputs(sample_name, df):
    sample_output_dir = os.path.join(LABELED_OUTPUT_DIR, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    df.to_excel(os.path.join(sample_output_dir, "metrics.xlsx"), index=False)
    df_slices = df[df["Slice"] != "MEAN"].copy()
    df_slices["Slice"] = df_slices["Slice"].astype(int)

    # Dice plot
    plt.figure()
    plt.plot(df_slices["Slice"], df_slices["Dice"])
    plt.title("Dice per Slice")
    plt.savefig(os.path.join(sample_output_dir, "dice_per_slice.png"))
    plt.close()

    # IoU plot
    plt.figure()
    plt.plot(df_slices["Slice"], df_slices["IoU"])
    plt.title("IoU per Slice")
    plt.savefig(os.path.join(sample_output_dir, "iou_per_slice.png"))
    plt.close()

    # Average bar plot
    mean_row = df[df["Slice"] == "MEAN"].iloc[0]
    plt.figure()
    plt.bar(
        ["Dice", "IoU", "Precision", "Recall", "Accuracy"],
        [
            mean_row["Dice"],
            mean_row["IoU"],
            mean_row["Precision"],
            mean_row["Recall"],
            mean_row["Accuracy"]])
    plt.ylim(0, 1)
    plt.title("Average Metrics")
    plt.savefig(os.path.join(sample_output_dir, "average_metrics.png"))
    plt.close()
    return mean_row

def save_combined_results(all_sample_means):
    combined_dir = os.path.join(LABELED_OUTPUT_DIR, "combined_results")
    os.makedirs(combined_dir, exist_ok=True)
    combined_df = pd.DataFrame(all_sample_means)
    overall_mean = combined_df.mean(numeric_only=True)
    overall_row = overall_mean.copy()
    overall_row["Slice"] = "OVERALL_MEAN"
    overall_row["Sample"] = "ALL"
    combined_df = pd.concat([combined_df, pd.DataFrame([overall_row])],ignore_index=True)
    combined_df.to_excel(os.path.join(combined_dir, "combined_metrics.xlsx"), index=False)
    print("\nOverall Mean Across All Samples:")
    print(overall_mean)

    # Overall Average Bar Plot
    plt.figure()
    plt.bar(
        ["Dice", "IoU", "Precision", "Recall", "Accuracy"],
        [
            overall_mean["Dice"],
            overall_mean["IoU"],
            overall_mean["Precision"],
            overall_mean["Recall"],
            overall_mean["Accuracy"]])
    plt.ylim(0, 1)
    plt.title("Overall Average Across All Samples")
    plt.savefig(os.path.join(combined_dir, "overall_average_bar.png"))
    plt.close()

    # Sample-wise Dice Comparison
    plt.figure()
    plt.plot(combined_df["Sample"][:-1], combined_df["Dice"][:-1], marker="o")
    plt.xticks(rotation=45)
    plt.title("Dice Comparison Across Samples")
    plt.ylabel("Dice")
    plt.savefig(os.path.join(combined_dir, "dice_across_samples.png"))
    plt.close()

    #Boxplot
    plt.figure()
    plt.boxplot([
        combined_df["Dice"][:-1],
        combined_df["IoU"][:-1],
        combined_df["Precision"][:-1],
        combined_df["Recall"][:-1],
        combined_df["Accuracy"][:-1]])
    plt.xticks([1, 2, 3, 4, 5],["Dice", "IoU", "Precision", "Recall", "Accuracy"])
    plt.title("Metric Distribution Across Samples")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(combined_dir, "metric_distribution_boxplot.png"))
    plt.close()

    #Heatmap
    plt.figure()
    data_matrix = combined_df[:-1][["Dice", "IoU", "Precision", "Recall", "Accuracy"]].values
    plt.imshow(data_matrix)
    plt.xticks(range(5),["Dice", "IoU", "Precision", "Recall", "Accuracy"])
    plt.yticks(range(len(combined_df[:-1])), combined_df["Sample"][:-1])
    plt.colorbar()
    plt.title("Metric Heatmap Across Samples")
    plt.savefig(os.path.join(combined_dir, "metric_heatmap.png"))
    plt.close()

def main():
    os.makedirs(LABELED_OUTPUT_DIR, exist_ok=True)
    label_files = sorted(
        f for f in os.listdir(MANUAL_LABEL_DIR)
        if f.endswith("_label.tif"))
    print(f"\nFound {len(label_files)} labeled samples.\n")
    all_sample_means = []
    for label_file in label_files:
        sample_name = label_file.replace("_label.tif", "")
        df = evaluate_sample(sample_name)
        if df is None:
            continue
        mean_row = save_sample_outputs(sample_name, df)
        mean_row["Sample"] = sample_name
        all_sample_means.append(mean_row)
    if all_sample_means:
        save_combined_results(all_sample_means)
    print("\nEvaluation completed for all labeled samples.")

if __name__ == "__main__":
    main()