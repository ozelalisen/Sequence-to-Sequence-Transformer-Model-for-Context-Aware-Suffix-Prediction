import random
import json
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from uuid import uuid4

# import pm4py
from tensorflow.keras.losses import sparse_categorical_crossentropy
from sklearn import metrics

from processtransformer import constants
from processtransformer.models.transformer import Transformer
from processtransformer.data.loader import LogsDataLoader
from processtransformer.data.processor import LogsDataProcessor

import warnings

warnings.filterwarnings("ignore")


def levenshtein_similarity(seq1, seq2):
    return 1 - (editdistance.eval(seq1, seq2) / max(len(seq1), len(seq2)))


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def xes_to_csv(dataset_path, file_name):
    import pm4py

    log = pm4py.read_xes(dataset_path)
    pd = pm4py.convert_to_dataframe(log)
    pd.to_csv(file_name, index=False)


datasets = {
    "bpi_challenge_2020_travel_permit": {
        "dataset_path": "bpi_2020_permit_log.csv",
        "columns": [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        "context_columns": [
            "case:OrganizationalEntity",
            "org:role",
            "org:resource",
        ],
        "cost_column": "case:TotalDeclared",
    },
    # "bpi_challenge_2020_domestic": {
    #     "dataset_path": "bpi_2020_domestic_declarations.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": [
    #         "org:resource",
    #         "org:role",
    #     ],
    #     "cost_column": "case:Amount",
    # },
    # "bpi_challenge_2020_international": {
    #     "dataset_path": "bpi_2020_international_declarations.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": [
    #         "org:resource",
    #         "org:role",
    #         "case:Permit OrganizationalEntity",
    #     ],
    #     "cost_column": "case:Amount",
    # },
    # "bpi_challenge_2020_prepaid_travel_cost": {
    #     "dataset_path": "bpi_2020_prepaid_travel_cost.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": [
    #         "case:OrganizationalEntity",
    #         "org:role",
    #         "case:Project",
    #     ],
    #     "cost_column": "case:RequestedAmount",
    # },
    # "bpi_challenge_2020_request_for_payment": {
    #     "dataset_path": "bpi_2020_request_for_payment.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": [
    #         "case:OrganizationalEntity",
    #         "org:role",
    #         "case:Project",
    #     ],
    #     "cost_column": "case:RequestedAmount",
    # },
    # "bpi_challenge_2012": {
    #     "dataset_path": "BPI_Challenge_2012.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": ["org:resource", "lifecycle:transition"],
    #     "cost_column": "case:AMOUNT_REQ",
    # },
    # "bpi_challenge_2017": {
    #     "dataset_path": "BPI_Challenge_2017.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": [
    #         "Action",
    #         "org:resource",
    #         "lifecycle:transition",
    #         "case:LoanGoal",
    #     ],
    #     "cost_column": "case:RequestedAmount",
    # },
    # "od_dms": {
    #     "dataset_path": "prod_od_dms_dataset.csv",
    #     "columns": [
    #         "case_id",
    #         "activity",
    #         "timestamp",
    #     ],
    #     "context_columns": [
    #         "od_system",
    #         "platform",
    #     ],
    #     "cost_column": "file_size",
    # },
    # "helpdesk": {
    #     "dataset_path": "Helpdesk.csv",
    #     "columns": [
    #         "Case ID",
    #         "Activity",
    #         "Complete Timestamp",
    #     ],
    #     "context_columns": [
    #         "Resource",
    #         # "workgroup",
    #         # "customer",
    #         # "seriousness",
    #         # "product",
    #         # "seriousness_2",
    #         # "service_level",
    #         # "service_type",
    #         # "support_section",
    #         # "responsible_section",
    #     ],
    # },
    # "sepsis": {
    #     "dataset_path": "sepsis.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": ["org:group"],
    #     "cost_column": "Leucocytes",
    # },
    # "road_traffic_fine_management_process": {
    #     "dataset_path": "Road_Traffic_Fine_Management_Process.csv",
    #     "columns": [
    #         "case:concept:name",
    #         "concept:name",
    #         "time:timestamp",
    #     ],
    #     "context_columns": [
    #         "org:resource",
    #         "dismissal",
    #         "article",
    #     ],
    #     "cost_column": "amount",
    # },
}


# def masked_loss(label, pred):
#     mask = label != 0
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#         from_logits=True, reduction="none"
#     )
#     loss = loss_object(label, pred)
#
#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask
#
#     loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
#     return loss
#
#
# def masked_accuracy(label, pred):
#     pred = tf.argmax(pred, axis=2)
#     label = tf.cast(label, pred.dtype)
#     match = label == pred
#
#     mask = label != 0
#
#     match = match & mask
#
#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match) / tf.reduce_sum(mask)
#


def process_dataset(dataset_name: str):
    dataset_doc = datasets[dataset_name]
    data_processor = LogsDataProcessor(
        name=dataset_name,
        filepath=f"./datasets/{dataset_doc['dataset_path']}",
        columns=dataset_doc["columns"]
        + [dataset_doc["cost_column"]]
        + dataset_doc["context_columns"],
        dir_path="datasets",
        pool=4,
    )
    data_processor.process_logs(
        task=constants.Task.NEXT_ACTIVITY, sort_temporally=False
    )


def evaluate_next_activity_with_context(
    dataset_name: str,
    context_columns: List[str],
    regression_column: str,
    prefix_length: str = "Max",
    n_splits: int = 5,
) -> None:
    data_loader = LogsDataLoader(name=dataset_name)
    (
        all_data_df,
        x_word_dict,
        context_dict,
        max_case_length,
        vocab_size,
        context_vocab_sizes,
        num_output,
        input_dimension,
    ) = data_loader.load_data(
        constants.Task.NEXT_ACTIVITY, regression_column, context_columns, prefix_length
    )

    print(f"Max case length is {max_case_length}")

    kf = KFold(n_splits=n_splits, shuffle=False)

    learning_rate = 0.001
    batch_size = 256
    epochs = 10

    # Define lists to store evaluation metrics across all folds
    (
        all_accuracies,
        all_fscores,
        all_precisions,
        all_recalls,
        all_l_similarities,
        all_maes,
        all_mses,
        all_rmses,
        all_len_test_indexes,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    training_times = []
    fold = 0
    for train_index, test_index in kf.split(all_data_df):
        fold += 1
        train_df = all_data_df.iloc[train_index]
        test_df = all_data_df.iloc[test_index]

        # Prepare training examples for next activity prediction task
        (
            train_token_x,
            train_decoder_x,
            train_token_context,
            train_token_y,
            train_token_y_cost,
            cost_scaler,
        ) = data_loader.prepare_data_next_activity(
            train_df, x_word_dict, context_dict, max_case_length, regression_column
        )

        # Create and compile a transformer model

        transformer_model = Transformer(
            max_len=max_case_length,
            input_vocab_size=vocab_size,
            target_vocab_size=num_output,
            context_vocab_sizes=context_vocab_sizes,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        transformer_model.compile(
            optimizer=optimizer,
            loss={
                "suffix_activity": masked_loss,
                "suffix_file_size": tf.keras.losses.MeanAbsoluteError(),
            },
            metrics={
                "suffix_activity": [masked_accuracy],
                "suffix_file_size": [tf.keras.metrics.MeanAbsoluteError()],
            },
        )

        inputs = (train_token_x, train_decoder_x, train_token_context)

        t0 = time.time()
        # Train the model
        transformer_model.fit(
            inputs,
            {
                "suffix_activity": train_token_y,
                "suffix_file_size": train_token_y_cost,
            },
            epochs=epochs,
            batch_size=batch_size,
        )
        training_time = time.time() - t0
        training_times.append(training_time)

        # Evaluate over all the prefixes (k) and save the results
        (
            k,
            accuracies,
            fscores,
            precisions,
            recalls,
            l_similarities,
            len_test_indexes,
            maes,
            mses,
            rmses,
        ) = ([], [], [], [], [], [], [], [], [], [])
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"] == i]
            if len(test_data_subset) > 0:
                (
                    test_token_x,
                    test_token_decoder,
                    test_token_context,
                    test_token_y,
                    test_token_y_cost,
                    test_cost_scaler,
                ) = data_loader.prepare_data_next_activity(
                    test_data_subset,
                    x_word_dict,
                    context_dict,
                    max_case_length,
                    regression_column,
                    cost_scaler,
                )

                y_pred, file_size_pred, attention_scores = predict_sequences(
                    transformer_model,
                    test_token_x,
                    test_token_decoder,
                    test_token_context,
                    max_case_length,
                    i,
                )

                similarities = []
                for pred_seq, true_seq in zip(y_pred, test_token_y):
                    pred_seq = unpad_trailing_zeros_row(pred_seq)
                    true_seq = unpad_trailing_zeros_row(true_seq)
                    similarity = levenshtein_similarity(pred_seq, true_seq)
                    similarities.append(similarity)

                l_similarity = np.mean(similarities)

                next_activity_pred = y_pred[:, 0]
                next_activity_y = test_token_y[:, 0]
                accuracy = metrics.accuracy_score(next_activity_pred, next_activity_y)
                precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                    next_activity_pred, next_activity_y, average="weighted"
                )

                k.append(i)
                len_test_indexes.append(len(test_data_subset))
                accuracies.append(accuracy)
                fscores.append(fscore)
                precisions.append(precision)
                recalls.append(recall)
                l_similarities.append(l_similarity)
                test_token_y_cost = cost_scaler.inverse_transform(test_token_y_cost)
                file_size_pred = cost_scaler.inverse_transform(file_size_pred)

                maes.append(
                    metrics.mean_absolute_error(
                        test_token_y_cost,
                        file_size_pred,
                    )
                )
                mses.append(
                    metrics.mean_squared_error(test_token_y_cost, file_size_pred)
                )
                rmses.append(
                    np.sqrt(
                        metrics.mean_squared_error(test_token_y_cost, file_size_pred)
                    )
                )

                if context_columns and i < 4:

                    random_index_for_plotting = random.choice(
                        range(0, len(test_data_subset))
                    )
                    try:
                        (
                            plot_token_x,
                            plot_token_decoder,
                            plot_token_context,
                            plot_token_y,
                            plot_token_y_cost,
                            plot_cost_scaler,
                        ) = data_loader.prepare_data_next_activity(
                            test_data_subset.iloc[[random_index_for_plotting]],
                            x_word_dict,
                            context_dict,
                            max_case_length,
                            regression_column,
                            cost_scaler,
                        )
                    except ValueError:
                        continue

                    plot_remaining_trace, _, plot_attention_scores = predict_sequences(
                        transformer_model,
                        plot_token_x,
                        plot_token_decoder,
                        plot_token_context,
                        max_case_length,
                        i,
                    )

                    plot_reversed_x_word_dict = {v: k for k, v in x_word_dict.items()}

                    plot_reversed_context_dict = {
                        outer_key: {v: k for k, v in inner_dict.items()}
                        for outer_key, inner_dict in context_dict.items()
                    }

                    plot_token_x_trimmed = unpad_trailing_zeros_row(plot_token_x[0])
                    plot_token_y_trimmed = unpad_trailing_zeros_row(
                        plot_remaining_trace[0]
                    )

                    plot_token_context_trimmed = [
                        unpad_trailing_zeros_row(token_context[0])
                        for token_context in plot_token_context
                    ]

                    plot_input_tokens = []

                    main_tokens = [
                        plot_reversed_x_word_dict.get(token, "<UNK>")
                        for token in plot_token_x_trimmed
                    ]
                    plot_input_tokens.append(main_tokens)

                    for j, token_context in enumerate(plot_token_context_trimmed):
                        context_mapping = plot_reversed_context_dict[
                            context_columns[j] + "_word_dict"
                        ]
                        plot_input_tokens.append(
                            [
                                context_mapping.get(token, "<UNK>")
                                for token in token_context
                            ]
                        )

                    plot_output_tokens = [
                        plot_reversed_x_word_dict.get(token, "<UNK>")
                        for token in plot_token_y_trimmed
                    ]

                    plot_attention_weights(
                        plot_input_tokens,
                        plot_output_tokens,
                        tf.squeeze(plot_attention_scores, 0),
                        max_case_length,
                        filename=f"plots/attention/{dataset_name}_{i}_{fold}_attention",
                    )

            else:
                k.append(i)
                len_test_indexes.append(0)
                accuracies.append(0)
                fscores.append(0)
                precisions.append(0)
                recalls.append(0)
                l_similarities.append(0)
                maes.append(0)
                mses.append(0)
                rmses.append(0)
        k.append(i + 1)
        accuracies.append(np.average(accuracies, weights=len_test_indexes))
        fscores.append(np.average(fscores, weights=len_test_indexes))
        precisions.append(np.average(precisions, weights=len_test_indexes))
        recalls.append(np.average(recalls, weights=len_test_indexes))
        l_similarities.append(np.average(l_similarities, weights=len_test_indexes))
        maes.append(np.average(maes, weights=len_test_indexes))
        mses.append(np.average(mses, weights=len_test_indexes))
        rmses.append(np.average(rmses, weights=len_test_indexes))
        len_test_indexes.append(np.sum(len_test_indexes))

        # Append metrics for this fold to the overall lists
        all_accuracies.append(accuracies)
        all_fscores.append(fscores)
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_l_similarities.append(l_similarities)
        all_maes.append(maes)
        all_mses.append(mses)
        all_rmses.append(rmses)
        all_len_test_indexes.append(len_test_indexes)
        print(f"Overall Accuracy {accuracies[-1]}")
        print(f"Overall Levenshtein Similarity {l_similarities[-1]}")
        print(f"Overall Mean Absolute Error {maes[-1]}")

    # Compute average performance metrics across all folds
    index_scores = {}
    for i in range(max_case_length + 1):
        weights = [len_test_indexes[i] for len_test_indexes in all_len_test_indexes]

        if np.sum(weights) == 0:
            accuracy = precision = fscore = recall = maes = mses = rmses = l_sim = 0.0
        else:
            accuracy = np.average(
                [accuracy[i] for accuracy in all_accuracies], weights=weights
            )
            precision = np.average(
                [precision[i] for precision in all_precisions], weights=weights
            )
            fscore = np.average([fscore[i] for fscore in all_fscores], weights=weights)
            recall = np.average([recall[i] for recall in all_recalls], weights=weights)
            maes = np.average([maes[i] for maes in all_maes], weights=weights)
            mses = np.average([mses[i] for mses in all_mses], weights=weights)
            rmses = np.average([rmses[i] for rmses in all_rmses], weights=weights)
            l_sim = np.average(
                [l_similarities[i] for l_similarities in all_l_similarities],
                weights=weights,
            )

        index_scores[str(i + 1)] = {
            "size": int(np.sum(weights)),
            "accuracy": float(accuracy),
            "fscore": float(fscore),
            "precision": float(precision),
            "recall": float(recall),
            "maes": float(maes),
            "mses": float(mses),
            "rmses": float(rmses),
            "levensthein_similarity": float(l_sim),
        }

    results = {}
    results.update(
        {
            str(uuid4()): {
                "dataset_name": dataset_name,
                "context columns": context_columns,
                "context_size": len(context_columns),
                "training_time": np.mean(training_times),
                "index_scores": index_scores,
                "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "n_splits": n_splits,
                "prefix_length": input_dimension,
                "model": "seq2seq Transformer",
            }
        }
    )
    pprint(results)
    current_results = {}
    if Path("results.json").is_file():
        with open("results.json", "r") as f:
            current_results = json.load(f)
    current_results.update(results)
    with open("results.json", "w+") as f:
        json.dump(
            current_results,
            f,
            indent=4,
            sort_keys=True,
        )


def predict_sequences(
    model,
    input_data,
    decoder_data,
    context_data,
    max_case_length,
    k=1,
    beam_width=3,
    end_token=3,
    search_strategy="greedy",  # "greedy" or "beam"
):
    # decoder_data[:, 1:] = 0
    remaining_trace = np.zeros_like(decoder_data)
    file_size_pred = np.zeros_like(decoder_data)
    batch_size = input_data.shape[0]

    for i in range(max_case_length - k - 1):
        inputs = (input_data, decoder_data, context_data)
        predictions = model.predict(inputs)
        file_size_pred = np.squeeze(predictions["suffix_file_size"], axis=-1)

        if search_strategy == "greedy":
            remaining_trace = np.argmax(predictions["suffix_activity"], axis=2)
        elif search_strategy == "beam":
            # Create a sequence_length vector where each entry is the current length i+1
            sequence_length = np.full((batch_size,), max_case_length, dtype=np.int32)

            decoded, log_prob = tf.nn.ctc_beam_search_decoder(
                tf.transpose(predictions["suffix_activity"], (1, 0, 2)),
                sequence_length=sequence_length,
                beam_width=beam_width,
            )
            # Convert the sparse tensor output to dense, using end_token as the placeholder
            decoded_dense = tf.sparse.to_dense(
                decoded[0], default_value=end_token
            ).numpy()

            next_tokens = np.full((batch_size,), end_token)  # Initialize with end_token
            for b in range(batch_size):
                if (
                    i < decoded_dense.shape[1]
                ):  # Check if the current index is within bounds
                    if decoded_dense[b, i] != end_token:
                        next_tokens[b] = decoded_dense[b, i]
                    else:
                        next_tokens[b] = (
                            end_token  # Set end_token if out of bounds or default value
                        )

        if i < max_case_length - 1:
            decoder_data[:, i + 1] = next_tokens
        remaining_trace[:, i] = next_tokens

        if np.all(next_tokens == end_token):
            break

    remaining_trace = zero_after_first_three(remaining_trace)
    mask = remaining_trace != 0
    file_size_pred[~mask] = 0

    return remaining_trace, file_size_pred, predictions["attn_scores"]


def unpad_trailing_zeros_row(row):
    non_zero_positions = np.where(row != 0)[0]
    if len(non_zero_positions) > 0:
        last_non_zero_position = non_zero_positions[-1]
        return row[: last_non_zero_position + 1]
    else:
        return row[:0]  # Return an empty row if all elements are zero


def zero_after_first_three(array):
    def process_row(row):
        indices = np.where(row == 3)[0]
        if len(indices) > 0:
            first_three_idx = indices[0]
            row[first_three_idx + 1 :] = 0
        return row

    return np.array([process_row(row) for row in array])


def plot_attention_head(in_tokens, translated_tokens, attention, ax, max_case_length):
    # Initialize a list to hold the non-padded sections of the attention matrix
    non_padded_attention = []
    xticks_labels = []

    # Loop over each case in in_tokens and extract the corresponding part of the attention matrix
    for i, tokens in enumerate(in_tokens):
        start_index = i * max_case_length
        end_index = start_index + len(tokens)
        non_padded_attention.append(attention[:, start_index:end_index])
        xticks_labels.extend(tokens)

    # Concatenate the non-padded sections together horizontally
    non_padded_attention = np.concatenate(non_padded_attention, axis=1)

    # Truncate non_padded_attention to match translated_tokens length
    non_padded_attention = non_padded_attention[: len(translated_tokens), :]

    # Plot the attention weights
    cax = ax.matshow(non_padded_attention, cmap="viridis")

    # Generate tick positions and labels for non-padded tokens
    xticks_positions = np.arange(non_padded_attention.shape[1])

    # Ensure the number of ticks matches the number of labels
    ax.set_xticks(xticks_positions)
    ax.set_yticks(range(len(translated_tokens)))

    # Set labels with actual token names
    ax.set_xticklabels(xticks_labels, rotation=90, fontsize=10)
    ax.set_yticklabels(translated_tokens, fontsize=10)

    return cax


def plot_attention_weights(
    input_tokens,
    output_tokens,
    attention_heads,
    max_case_length,
    filename="attention_plot",
):
    for h, head in enumerate(attention_heads):
        # Create a new figure for each attention head
        fig, ax = plt.subplots(figsize=(16, 12))  # Adjust size as needed

        # Plot the attention head
        cax = plot_attention_head(
            input_tokens, output_tokens, head, ax, max_case_length
        )
        ax.set_xlabel(f"Head {h+1}")

        # Add a colorbar
        fig.colorbar(cax, ax=ax, orientation="vertical")

        # Improve layout and aesthetics
        plt.tight_layout()

        # Save the figure
        fig.savefig(f"{filename}_head_{h+1}.png", dpi=300)
        plt.close(fig)  # Close the figure to avoid displaying it in the loop


def main():
    for name, doc in datasets.items():
        # process_dataset(name)
        print(f"Running dataset {name} without contextual data")
        evaluate_next_activity_with_context(name, [], doc["cost_column"])
        context_columns = doc["context_columns"]
        print(f"Running dataset {name} with {context_columns}")
        evaluate_next_activity_with_context(name, context_columns, doc["cost_column"])


if __name__ == "__main__":
    # xes_to_csv(
    #     "/home/oa/dms_projects/context-net/datasets/PermitLog.xes.gz",
    #     "datasets/bpi_2020_permit_log.csv",
    # )
    # process_dataset("bpi_challenge_2020_travel_permit")
    main()
