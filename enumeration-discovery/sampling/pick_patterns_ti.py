import pick_patterns as base


def select_ti_balanced_centers(embs, graph_list, target_num, negative_ratio):
    """Keep equal numbers of negative- and positive-centred TI subgraphs."""

    target_num = min(target_num, len(graph_list))
    negative = [index for index, graph in enumerate(graph_list) if base._center_label_id(graph) == 1]
    positive = [index for index, graph in enumerate(graph_list) if base._center_label_id(graph) == 2]

    def choose(indices, count):
        if not indices or count <= 0:
            return []
        local = base.select_graphs(
            embs=embs[indices], method=base.SELECTOR, k=min(count, len(indices)),
            seed=base.seed, sigma=base.PICK_SIGMA, chi=base.PICK_CHI,
        )
        return [indices[int(index)] for index in local]

    per_label = target_num // 2
    selected = choose(negative, per_label)
    selected.extend(choose(positive, per_label))
    selected_set = set(selected)
    remainder = [index for index in range(len(graph_list)) if index not in selected_set]
    selected.extend(choose(remainder, target_num - len(selected)))
    print(
        f"[SelectTI] raw_negative_centered={len(negative)} raw_positive_centered={len(positive)} "
        f"selected_negative_centered={sum(index in negative for index in selected)} "
        f"selected_positive_centered={sum(index in positive for index in selected)}"
    )
    return selected


if __name__ == "__main__":
    base.NEGATIVE_CENTER_RATIO = 0.5
    base.SELECTED_NEGATIVE_CENTER_RATIO = 0.5
    base.select_graphs_with_negative_quota = select_ti_balanced_centers
    base.main(["--relation", "ti"])
