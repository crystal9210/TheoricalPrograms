import networkx as nx
import matplotlib.pyplot as plt

# スモールワールドネットワークの生成
def create_small_world_network():
    n = 30  # ノード数
    k = 4   # 各ノードが接続する近傍ノード数
    p = 0.1 # 再配線確率

    G = nx.watts_strogatz_graph(n, k, p)
    return G

# スケールフリーネットワークの生成
def create_scale_free_network():
    n = 30  # ノード数
    G = nx.barabasi_albert_graph(n, 2)
    return G

# ネットワークのプロット
def plot_network(G, title, ax):
    pos = nx.spring_layout(G, seed=42)  # ノード配置の決定
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray', ax=ax)
    ax.set_title(title)

# メイン関数
def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # スモールワールドネットワークの生成とプロット
    G_small_world = create_small_world_network()
    plot_network(G_small_world, "Small-World Network", ax1)

    # スケールフリーネットワークの生成とプロット
    G_scale_free = create_scale_free_network()
    plot_network(G_scale_free, "Scale-Free Network", ax2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
