import matplotlib.pyplot as plt
from pathlib import Path

def plot_price_and_u(
    df,
    slot,
    price_col="price_scaled",
    save=False,
    save_dir="./fig",
):
    df_slot = df[df["slot"] == slot]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 上：価格
    axes[0].plot(df_slot.index, df_slot[price_col])
    axes[0].set_title(f"Price (slot={slot})")
    axes[0].set_ylabel("price")
    axes[0].grid(True)

    # 下：u（スパイク）
    axes[1].scatter(
        df_slot.index,
        df_slot["u"],
        s=10
    )
    axes[1].set_title("Spike indicator u")
    axes[1].set_ylabel("u")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True)

    axes[1].set_xlabel("datetime")

    plt.tight_layout()

    # 保存処理（フラグ制御）
    if save:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / f"price_plt_slot_{slot}.png"
        plt.savefig(file_path)

    plt.show()