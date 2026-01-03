import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from recycla import ROOT_PATH, log, DATA_PATH
from recycla.train.ModelType import ModelType
from recycla.train.train import _train

log.setLevel("WARNING")


# GPU Memory Management Utilities
def clear_gpu_memory():
    """Clear GPU memory cache and force garbage collection."""
    if torch.cuda.is_available():
        print("🧹 Clearing GPU memory...")

        # Clear the GPU cache
        torch.cuda.empty_cache()

        # Force garbage collection
        import gc

        gc.collect()

    else:
        print("ℹ️  No CUDA GPU available")


# Set data directory
data_dir = DATA_PATH / "labeled_data"


def run_representative_model_experiments():
    """Run training experiments with one representative model from each network architecture type."""

    results = []

    # Select one representative model from each architecture family
    representative_models = [
        (ModelType.MOBILENET_V2, "MobileNet V2", "Lightweight mobile architecture"),
        (
            ModelType.EFFICIENTNET_V2_S,
            "EfficientNet V2-S",
            "Efficient scaling architecture (small)",
        ),
        (
            ModelType.CONVNEXT_TINY,
            "ConvNeXt Tiny",
            "Modern convolutional architecture (tiny)",
        ),
        (
            ModelType.REGNET_Y_400MF,
            "RegNet Y-400MF",
            "Regularized efficient network (Y-400MF)",
        ),
        (
            ModelType.VIT_B_16,
            "Vision Transformer B-16",
            "Transformer-based vision model (base, 16x16 patches)",
        ),
        (ModelType.RESNET_18, "ResNet-18", "Classic residual network (18 layers)"),
    ]

    print(
        f"🚀 Starting representative model experiments at {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"📊 Total models to test: {len(representative_models)}")
    print("=" * 80)

    for i, (model_type, model_name, model_desc) in enumerate(representative_models, 1):
        print(f"\n🔬 Experiment {i}/{len(representative_models)}: {model_name}")
        print(f"🏗️  Architecture: {model_desc}")
        print(f"🔧 Model Type: {model_type.value}")

        clear_gpu_memory()
        gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"💾 GPU Memory Allocated: {gpu_memory:.2f} MB")
        print(f"💾 GPU Memory Reserved: {gpu_reserved:.2f} MB")

        start_time = time.time()

        try:
            # Run training with current model type using default augmentations
            (
                train_losses,
                train_primary_accs,
                train_secondary_accs,
                val_losses,
                val_primary_accs,
                val_secondary_accs,
            ) = _train(
                data_dir=data_dir,
                nepochs=10,  # Reduced epochs for faster testing
                checkpoint_path=None,
                weight_labels=False,
                transform_kwargs=None,  # Use default augmentations
                model_type=model_type,
            )

            elapsed_time = time.time() - start_time

            # Move the saved model to the new location
            run_uid = str(uuid.uuid4())[:8]  # Generate a unique run ID
            description = f"rep_{model_type.value}"

            # Create source and destination paths
            source_path = ROOT_PATH / ".models/best_candidate.pth"
            dest_dir = ROOT_PATH / "workspace/.models" / run_uid
            dest_path = dest_dir / f"{description}.pth"

            # Create destination directory if it doesn't exist
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Move the model file if it exists
            if source_path.exists():
                shutil.move(str(source_path), str(dest_path))
                print(f"📁 Model moved to: {dest_path}")
            else:
                print(f"⚠️  Model file not found at: {source_path}")

            # Find epoch with best validation loss
            best_loss_idx = np.argmin(val_losses)
            best_loss = val_losses[best_loss_idx]

            # Get accuracies at epoch of best validation loss
            accuracy_at_best_loss_primary = val_primary_accs[best_loss_idx]
            accuracy_at_best_loss_secondary = val_secondary_accs[best_loss_idx]

            # Also track final epoch accuracies for comparison
            final_primary_acc = val_primary_accs[-1]
            final_secondary_acc = val_secondary_accs[-1]

            result = {
                "model_type": model_type,
                "model_name": model_name,
                "model_desc": model_desc,
                "model_value": model_type.value,
                "best_loss": best_loss,
                "best_loss_idx": best_loss_idx,
                "accuracy_at_best_loss_primary": accuracy_at_best_loss_primary,
                "accuracy_at_best_loss_secondary": accuracy_at_best_loss_secondary,
                "final_primary_accuracy": final_primary_acc,
                "final_secondary_accuracy": final_secondary_acc,
                "training_time": elapsed_time,
                "status": "success",
                "run_uid": run_uid,
                "model_path": str(dest_path),
                # Store all training curves for plotting later
                "train_losses": train_losses,
                "train_primary_accs": train_primary_accs,
                "train_secondary_accs": train_secondary_accs,
                "val_losses": val_losses,
                "val_primary_accs": val_primary_accs,
                "val_secondary_accs": val_secondary_accs,
            }

            print(f"✅ Completed in {elapsed_time:.1f}s")
            print(f"📈 Best Loss: {best_loss:.4f} (epoch {best_loss_idx + 1})")
            print(
                f"🎯 Primary Accuracy @ Best Loss: {accuracy_at_best_loss_primary:.4f}"
            )
            print(
                f"🎯 Secondary Accuracy @ Best Loss: {accuracy_at_best_loss_secondary:.4f}"
            )
            print(f"📊 Final Primary Accuracy: {final_primary_acc:.4f}")
            print(f"📊 Final Secondary Accuracy: {final_secondary_acc:.4f}")

        except Exception as e:
            elapsed_time = time.time() - start_time

            # Get detailed error information
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

            # Find the most relevant line
            relevant_line = None
            line_number = None

            for tb_line in reversed(tb_lines):
                if 'File "' in tb_line and (
                    "train.py" in tb_line
                    or "loader_utils.py" in tb_line
                    or "<ipython-input-" in tb_line
                ):
                    try:
                        if "line " in tb_line:
                            line_part = tb_line.split("line ")[1].split(",")[0]
                            line_number = int(line_part)
                            relevant_line = tb_line.strip()
                            break
                    except:
                        pass

            error_msg = str(e)
            full_traceback = "".join(tb_lines)

            result = {
                "model_type": model_type,
                "model_name": model_name,
                "model_desc": model_desc,
                "model_value": model_type.value,
                "best_loss": float("inf"),
                "best_loss_idx": -1,
                "accuracy_at_best_loss_primary": 0.0,
                "accuracy_at_best_loss_secondary": 0.0,
                "final_primary_accuracy": 0.0,
                "final_secondary_accuracy": 0.0,
                "training_time": elapsed_time,
                "status": f"failed: {error_msg[:80]}",
                "error": error_msg,
                "line_number": line_number,
                "relevant_line": relevant_line,
                "full_traceback": full_traceback,
            }

            print(f"❌ Failed after {elapsed_time:.1f}s")
            print(f"🚨 Error: {error_msg}")
            if line_number:
                print(f"📍 Line: {line_number}")
            if relevant_line:
                print(f"📄 Location: {relevant_line}")
            print(f"🔍 Full traceback available in result['full_traceback']")

        results.append(result)
        print("-" * 60)

    return results


def print_experiment1_results(experiment1_results):
    # Print detailed results for Experiment 1
    print("\n" + "=" * 80)
    print("🏆 EXPERIMENT 1 RESULTS - REPRESENTATIVE MODEL ARCHITECTURES")
    print("=" * 80)

    # Sort results by accuracy at best loss (primary, then secondary as tiebreaker)
    successful_results = [r for r in experiment1_results if r["status"] == "success"]
    sorted_results = sorted(
        successful_results,
        key=lambda x: (
            x["accuracy_at_best_loss_primary"],
            x["accuracy_at_best_loss_secondary"],
        ),
        reverse=True,
    )

    print(f"\n📊 RANKING BY ACCURACY AT BEST VALIDATION LOSS:")
    print(
        f"{'Rank':<4} {'Model':<20} {'Architecture':<35} {'Loss':<8} {'Prim Acc':<9} {'Sec Acc':<9} {'Time':<8}"
    )
    print("-" * 110)

    for rank, result in enumerate(sorted_results, 1):
        print(
            f"{rank:<4} {result['model_name']:<20} {result['model_desc']:<35} "
            f"{result['best_loss']:<8.4f} {result['accuracy_at_best_loss_primary']:<9.4f} "
            f"{result['accuracy_at_best_loss_secondary']:<9.4f} {result['training_time']:<8.1f}s"
        )

    # Print error details for failed experiments
    failed_results = [r for r in experiment1_results if r["status"] != "success"]
    if failed_results:
        print(f"\n🚨 FAILED EXPERIMENTS DETAILS:")
        print("-" * 80)
        for result in failed_results:
            print(f"❌ {result['model_name']}:")
            print(f"   Error: {result['error']}")
            if result.get("line_number"):
                print(f"   Line: {result['line_number']}")
            if result.get("relevant_line"):
                print(f"   Location: {result['relevant_line']}")
            print()

    # Print analysis
    print(f"\n🔍 ARCHITECTURE FAMILY ANALYSIS:")

    if successful_results:
        best_model = max(
            successful_results, key=lambda x: x["accuracy_at_best_loss_primary"]
        )
        worst_model = min(
            successful_results, key=lambda x: x["accuracy_at_best_loss_primary"]
        )

        print(f"🥇 Best performing architecture: {best_model['model_name']}")
        print(f"   {best_model['model_desc']}")
        print(
            f"   Primary Accuracy @ Best Loss: {best_model['accuracy_at_best_loss_primary']:.4f}"
        )
        print(
            f"   Secondary Accuracy @ Best Loss: {best_model['accuracy_at_best_loss_secondary']:.4f}"
        )
        print(
            f"   Best Loss: {best_model['best_loss']:.4f} (epoch {best_model['best_loss_idx'] + 1})"
        )

        print(f"\n🥉 Worst performing architecture: {worst_model['model_name']}")
        print(f"   {worst_model['model_desc']}")
        print(
            f"   Primary Accuracy @ Best Loss: {worst_model['accuracy_at_best_loss_primary']:.4f}"
        )
        print(
            f"   Secondary Accuracy @ Best Loss: {worst_model['accuracy_at_best_loss_secondary']:.4f}"
        )

        avg_primary_acc = sum(
            r["accuracy_at_best_loss_primary"] for r in successful_results
        ) / len(successful_results)
        avg_secondary_acc = sum(
            r["accuracy_at_best_loss_secondary"] for r in successful_results
        ) / len(successful_results)
        avg_training_time = sum(r["training_time"] for r in successful_results) / len(
            successful_results
        )

        print(f"\n📊 Average metrics across architectures:")
        print(f"   Primary accuracy @ best loss: {avg_primary_acc:.4f}")
        print(f"   Secondary accuracy @ best loss: {avg_secondary_acc:.4f}")
        print(f"   Training time: {avg_training_time:.1f}s")

        # Architecture family performance
        print(f"\n🏗️  Architecture family rankings:")
        for i, result in enumerate(sorted_results, 1):
            efficiency_score = result["accuracy_at_best_loss_primary"] / (
                result["training_time"] / 60
            )  # accuracy per minute
            print(
                f"   {i}. {result['model_name']}: Acc {result['accuracy_at_best_loss_primary']:.4f}, "
                f"Time {result['training_time']:.1f}s, Efficiency {efficiency_score:.3f} acc/min"
            )

    else:
        print("❌ No successful experiments completed.")

    print(
        f"\n⏱️  Total experiment time: {sum(r['training_time'] for r in experiment1_results):.1f} seconds"
    )
    print(
        f"✅ Successful experiments: {len(successful_results)}/{len(experiment1_results)}"
    )
    print(f"❌ Failed experiments: {len(failed_results)}/{len(experiment1_results)}")

    # Store the best performing models for experiment 2 exclusion
    if successful_results:
        experiment1_model_types = [r["model_type"] for r in successful_results]
        print(f"\n💾 Results stored in 'experiment1_results' variable")
        print(
            f"📋 Model types tested: {[r['model_value'] for r in successful_results]}"
        )
    else:
        experiment1_model_types = []

    print("Experiment 1 completed! 🎉")
    return experiment1_model_types, successful_results


def run_comprehensive_model_experiments(exclude_model_types=None):
    """Run training experiments with all available model types, excluding specified ones."""

    if exclude_model_types is None:
        exclude_model_types = []

    results = []

    # Get all available model types, excluding those already tested
    all_model_types = [
        model_type for model_type in ModelType if model_type not in exclude_model_types
    ]

    print(
        f"🚀 Starting comprehensive model experiments at {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"📊 Total models to test: {len(all_model_types)}")
    print(
        f"🚫 Excluding {len(exclude_model_types)} models already tested in Experiment 1"
    )
    print("=" * 80)

    for i, model_type in enumerate(all_model_types, 1):
        model_name = model_type.value.replace("_", " ").title()

        print(f"\n🔬 Experiment {i}/{len(all_model_types)}: {model_name}")
        print(f"🔧 Model Type: {model_type.value}")

        start_time = time.time()

        try:
            # Run training with current model type using default augmentations
            (
                train_losses,
                train_primary_accs,
                train_secondary_accs,
                val_losses,
                val_primary_accs,
                val_secondary_accs,
            ) = _train(
                data_dir=data_dir,
                nepochs=10,  # Reduced epochs for faster testing
                checkpoint_path=None,
                weight_labels=False,
                transform_kwargs=None,  # Use default augmentations
                model_type=model_type,
            )

            elapsed_time = time.time() - start_time

            # Move the saved model to the new location
            run_uid = str(uuid.uuid4())[:8]  # Generate a unique run ID
            description = f"comp_{model_type.value}"

            # Create source and destination paths
            source_path = ROOT_PATH / ".models/best_candidate.pth"
            dest_dir = ROOT_PATH / "workspace/.models" / run_uid
            dest_path = dest_dir / f"{description}.pth"

            # Create destination directory if it doesn't exist
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Move the model file if it exists
            if source_path.exists():
                shutil.move(str(source_path), str(dest_path))
                print(f"📁 Model moved to: {dest_path}")
            else:
                print(f"⚠️  Model file not found at: {source_path}")

            # Find epoch with best validation loss
            best_loss_idx = np.argmin(val_losses)
            best_loss = val_losses[best_loss_idx]

            # Get accuracies at epoch of best validation loss
            accuracy_at_best_loss_primary = val_primary_accs[best_loss_idx]
            accuracy_at_best_loss_secondary = val_secondary_accs[best_loss_idx]

            # Also track final epoch accuracies for comparison
            final_primary_acc = val_primary_accs[-1]
            final_secondary_acc = val_secondary_accs[-1]

            result = {
                "model_type": model_type,
                "model_name": model_name,
                "model_value": model_type.value,
                "best_loss": best_loss,
                "best_loss_idx": best_loss_idx,
                "accuracy_at_best_loss_primary": accuracy_at_best_loss_primary,
                "accuracy_at_best_loss_secondary": accuracy_at_best_loss_secondary,
                "final_primary_accuracy": final_primary_acc,
                "final_secondary_accuracy": final_secondary_acc,
                "training_time": elapsed_time,
                "status": "success",
                "run_uid": run_uid,
                "model_path": str(dest_path),
                # Store all training curves for plotting later
                "train_losses": train_losses,
                "train_primary_accs": train_primary_accs,
                "train_secondary_accs": train_secondary_accs,
                "val_losses": val_losses,
                "val_primary_accs": val_primary_accs,
                "val_secondary_accs": val_secondary_accs,
            }

            print(f"✅ Completed in {elapsed_time:.1f}s")
            print(f"📈 Best Loss: {best_loss:.4f} (epoch {best_loss_idx + 1})")
            print(
                f"🎯 Primary Accuracy @ Best Loss: {accuracy_at_best_loss_primary:.4f}"
            )
            print(
                f"🎯 Secondary Accuracy @ Best Loss: {accuracy_at_best_loss_secondary:.4f}"
            )
            print(f"📊 Final Primary Accuracy: {final_primary_acc:.4f}")
            print(f"📊 Final Secondary Accuracy: {final_secondary_acc:.4f}")

        except Exception as e:
            elapsed_time = time.time() - start_time

            # Get detailed error information
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

            # Find the most relevant line
            relevant_line = None
            line_number = None

            for tb_line in reversed(tb_lines):
                if 'File "' in tb_line and (
                    "train.py" in tb_line
                    or "loader_utils.py" in tb_line
                    or "<ipython-input-" in tb_line
                ):
                    try:
                        if "line " in tb_line:
                            line_part = tb_line.split("line ")[1].split(",")[0]
                            line_number = int(line_part)
                            relevant_line = tb_line.strip()
                            break
                    except:
                        pass

            error_msg = str(e)
            full_traceback = "".join(tb_lines)

            result = {
                "model_type": model_type,
                "model_name": model_name,
                "model_value": model_type.value,
                "best_loss": float("inf"),
                "best_loss_idx": -1,
                "accuracy_at_best_loss_primary": 0.0,
                "accuracy_at_best_loss_secondary": 0.0,
                "final_primary_accuracy": 0.0,
                "final_secondary_accuracy": 0.0,
                "training_time": elapsed_time,
                "status": f"failed: {error_msg[:80]}",
                "error": error_msg,
                "line_number": line_number,
                "relevant_line": relevant_line,
                "full_traceback": full_traceback,
            }

            print(f"❌ Failed after {elapsed_time:.1f}s")
            print(f"🚨 Error: {error_msg}")
            if line_number:
                print(f"📍 Line: {line_number}")
            if relevant_line:
                print(f"📄 Location: {relevant_line}")
            print(f"🔍 Full traceback available in result['full_traceback']")

        results.append(result)
        print("-" * 60)

    return results


def analyze_and_plot_all_experiments(experiment2_results):
    """Provide comprehensive analysis and plotting for both experiments."""

    # Print Experiment 2 Results
    print("\n" + "=" * 80)
    print("🏆 EXPERIMENT 2 RESULTS - COMPREHENSIVE MODEL TESTING")
    print("=" * 80)

    successful_exp2 = [r for r in experiment2_results if r["status"] == "success"]
    sorted_exp2 = sorted(
        successful_exp2,
        key=lambda x: (
            x["accuracy_at_best_loss_primary"],
            x["accuracy_at_best_loss_secondary"],
        ),
        reverse=True,
    )

    print(f"\n📊 TOP 10 MODELS BY ACCURACY AT BEST VALIDATION LOSS:")
    print(
        f"{'Rank':<4} {'Model':<25} {'Loss':<8} {'Prim Acc':<9} {'Sec Acc':<9} {'Time':<8}"
    )
    print("-" * 75)

    for rank, result in enumerate(sorted_exp2[:10], 1):
        print(
            f"{rank:<4} {result['model_name']:<25} "
            f"{result['best_loss']:<8.4f} {result['accuracy_at_best_loss_primary']:<9.4f} "
            f"{result['accuracy_at_best_loss_secondary']:<9.4f} {result['training_time']:<8.1f}s"
        )

    # Combined analysis
    print(f"\n� COMBINED ANALYSIS - ALL EXPERIMENTS")
    print("=" * 80)

    # Combine results from both experiments
    all_successful = []
    if "experiment1_results" in locals():
        all_successful.extend(
            [r for r in experiment1_results if r["status"] == "success"]
        )
    all_successful.extend(successful_exp2)

    if not all_successful:
        print("❌ No successful experiments to analyze.")
        return

    # Sort all results
    all_sorted = sorted(
        all_successful, key=lambda x: x["accuracy_at_best_loss_primary"], reverse=True
    )

    print(f"\n🥇 OVERALL TOP 10 MODELS:")
    print(
        f"{'Rank':<4} {'Model':<25} {'Experiment':<8} {'Prim Acc':<9} {'Sec Acc':<9} {'Time':<8}"
    )
    print("-" * 80)

    for rank, result in enumerate(all_sorted[:10], 1):
        exp_label = "Exp1" if result in experiment1_results else "Exp2"
        print(
            f"{rank:<4} {result['model_name']:<25} {exp_label:<8} "
            f"{result['accuracy_at_best_loss_primary']:<9.4f} "
            f"{result['accuracy_at_best_loss_secondary']:<9.4f} {result['training_time']:<8.1f}s"
        )

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Comprehensive Model Architecture Comparison", fontsize=16)

    # Plot 1: Top 10 models - Primary Accuracy
    ax1 = axes[0, 0]
    top_10 = all_sorted[:10]
    model_names = [r["model_name"][:15] for r in top_10]  # Truncate names
    primary_accs = [r["accuracy_at_best_loss_primary"] for r in top_10]

    bars = ax1.barh(range(len(model_names)), primary_accs, alpha=0.8)
    ax1.set_yticks(range(len(model_names)))
    ax1.set_yticklabels(model_names)
    ax1.set_xlabel("Primary Accuracy @ Best Loss")
    ax1.set_title("Top 10 Models - Primary Accuracy")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, primary_accs)):
        ax1.text(acc + 0.001, i, f"{acc:.3f}", va="center", fontsize=8)

    # Plot 2: Architecture Family Comparison (Experiment 1)
    ax2 = axes[0, 1]
    if "experiment1_results" in locals():
        exp1_successful = [r for r in experiment1_results if r["status"] == "success"]
        if exp1_successful:
            families = [r["model_name"] for r in exp1_successful]
            primary_scores = [
                r["accuracy_at_best_loss_primary"] for r in exp1_successful
            ]
            secondary_scores = [
                r["accuracy_at_best_loss_secondary"] for r in exp1_successful
            ]

            x = range(len(families))
            width = 0.35

            ax2.bar(
                [i - width / 2 for i in x],
                primary_scores,
                width,
                label="Primary Accuracy",
                alpha=0.8,
            )
            ax2.bar(
                [i + width / 2 for i in x],
                secondary_scores,
                width,
                label="Secondary Accuracy",
                alpha=0.8,
            )

            ax2.set_xlabel("Architecture Family")
            ax2.set_ylabel("Accuracy @ Best Loss")
            ax2.set_title("Architecture Family Comparison (Exp 1)")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f[:10] for f in families], rotation=45, ha="right")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

    # Plot 3: Training Time vs Accuracy
    ax3 = axes[0, 2]
    times = [r["training_time"] for r in all_sorted[:15]]
    accs = [r["accuracy_at_best_loss_primary"] for r in all_sorted[:15]]
    colors = ["red" if r in experiment1_results else "blue" for r in all_sorted[:15]]

    scatter = ax3.scatter(times, accs, c=colors, alpha=0.7, s=50)
    ax3.set_xlabel("Training Time (seconds)")
    ax3.set_ylabel("Primary Accuracy @ Best Loss")
    ax3.set_title("Training Time vs Accuracy (Top 15)")
    ax3.grid(True, alpha=0.3)

    # Add legend for colors
    ax3.scatter([], [], c="red", alpha=0.7, s=50, label="Experiment 1")
    ax3.scatter([], [], c="blue", alpha=0.7, s=50, label="Experiment 2")
    ax3.legend()

    # Plot 4: Training curves for top 5 models
    ax4 = axes[1, 0]
    top_5 = all_sorted[:5]
    for result in top_5:
        epochs = range(1, len(result["val_losses"]) + 1)
        ax4.plot(
            epochs,
            result["val_losses"],
            "-",
            label=f"{result['model_name'][:12]}",
            alpha=0.8,
        )

        # Mark best validation loss
        best_idx = result["best_loss_idx"]
        ax4.scatter(
            best_idx + 1, result["best_loss"], s=100, marker="*", zorder=5, alpha=0.8
        )

    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation Loss")
    ax4.set_title("Training Curves - Top 5 Models")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Architecture family performance distribution
    ax5 = axes[1, 1]

    # Group models by architecture family
    family_groups = {
        "MobileNet": [r for r in all_successful if "mobilenet" in r["model_value"]],
        "EfficientNet": [
            r for r in all_successful if "efficientnet" in r["model_value"]
        ],
        "ConvNeXt": [r for r in all_successful if "convnext" in r["model_value"]],
        "RegNet": [r for r in all_successful if "regnet" in r["model_value"]],
        "ViT": [r for r in all_successful if "vit" in r["model_value"]],
        "ResNet": [r for r in all_successful if "resnet" in r["model_value"]],
    }

    family_names = []
    family_accs = []

    for family, models in family_groups.items():
        if models:
            family_names.append(family)
            accs = [m["accuracy_at_best_loss_primary"] for m in models]
            family_accs.append(accs)

    if family_accs:
        ax5.boxplot(family_accs, labels=family_names)
        ax5.set_ylabel("Primary Accuracy @ Best Loss")
        ax5.set_title("Architecture Family Performance Distribution")
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")

    # Plot 6: Model size vs performance (if we can infer size from names)
    ax6 = axes[1, 2]

    # Create efficiency score (accuracy per training time)
    efficiency_scores = [
        r["accuracy_at_best_loss_primary"] / (r["training_time"] / 60)
        for r in all_sorted[:15]
    ]
    model_names_short = [r["model_name"][:15] for r in all_sorted[:15]]

    bars = ax6.barh(range(len(model_names_short)), efficiency_scores, alpha=0.8)
    ax6.set_yticks(range(len(model_names_short)))
    ax6.set_yticklabels(model_names_short)
    ax6.set_xlabel("Efficiency (Accuracy per Minute)")
    ax6.set_title("Model Efficiency Ranking")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print comprehensive statistics
    print(f"\n📊 COMPREHENSIVE STATISTICS")
    print("-" * 50)

    best_overall = all_sorted[0]
    print(f"🥇 Best Overall Model: {best_overall['model_name']}")
    print(f"   Primary Accuracy: {best_overall['accuracy_at_best_loss_primary']:.4f}")
    print(
        f"   Secondary Accuracy: {best_overall['accuracy_at_best_loss_secondary']:.4f}"
    )
    print(f"   Training Time: {best_overall['training_time']:.1f}s")

    # Calculate family averages
    print(f"\n🏗️  Architecture Family Averages:")
    for family, models in family_groups.items():
        if models:
            avg_acc = sum(m["accuracy_at_best_loss_primary"] for m in models) / len(
                models
            )
            avg_time = sum(m["training_time"] for m in models) / len(models)
            print(
                f"   {family:12}: {avg_acc:.4f} accuracy, {avg_time:.1f}s avg time ({len(models)} models)"
            )

    # Find most efficient model
    most_efficient = max(
        all_successful,
        key=lambda x: x["accuracy_at_best_loss_primary"] / (x["training_time"] / 60),
    )
    efficiency_score = most_efficient["accuracy_at_best_loss_primary"] / (
        most_efficient["training_time"] / 60
    )
    print(f"\n⚡ Most Efficient Model: {most_efficient['model_name']}")
    print(f"   Efficiency Score: {efficiency_score:.3f} accuracy/minute")
    print(f"   Accuracy: {most_efficient['accuracy_at_best_loss_primary']:.4f}")
    print(f"   Training Time: {most_efficient['training_time']:.1f}s")

    print(
        f"\n⏱️  Total experimental time: {sum(r['training_time'] for r in all_successful):.1f} seconds"
    )
    print(f"📊 Total models tested: {len(all_successful)}")

    return all_sorted


@click.command()
def model_experiment():
    # Run the first experiment
    print("Starting representative model architecture testing...")
    experiment1_results = run_representative_model_experiments()

    experiment1_model_types, successful_results = print_experiment1_results(
        experiment1_results
    )

    experiment2_results = run_comprehensive_model_experiments(
        exclude_model_types=experiment1_model_types
    )

    final_rankings = analyze_and_plot_all_experiments(experiment2_results)
