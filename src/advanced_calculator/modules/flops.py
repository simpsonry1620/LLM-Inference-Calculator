from src.advanced_calculator.modules.utils import validate_positive_integer


class FLOPsCalculator:
    """
    Calculator for estimating FLOPs (Floating Point Operations) requirements for LLMs.
    """

    def __init__(self, history_callback=None):
        """
        Initialize FLOPs calculator

        Args:
            history_callback: Optional callback function to log calculation history
        """
        self._history_callback = history_callback

    def calculate_attention(
        self, batch_size: int, sequence_length: int, hidden_dimensions: int
    ) -> int:
        """
        Calculate FLOPs for attention mechanism.

        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model

        Returns:
            Estimated FLOPs for attention computation

        Formula:
            FLOPs_Attention = Batch_Size * Sequence_Length *
                             (Hidden_Dimensions^2 + Sequence_Length * Hidden_Dimensions)
        """
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(sequence_length, "Sequence length")
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")

        hidden_dim_squared = hidden_dimensions**2
        seq_len_times_hidden = sequence_length * hidden_dimensions
        result = (
            batch_size * sequence_length * (hidden_dim_squared + seq_len_times_hidden)
        )

        if self._history_callback:
            self._history_callback(
                f"FLOPs_Attention(batch_size={batch_size}, sequence_length={sequence_length}, "
                f"hidden_dimensions={hidden_dimensions}) = {result}"
            )
        return result

    def calculate_feedforward(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_dimensions: int,
        feedforward_dimensions: int,
    ) -> int:
        """
        Calculate FLOPs for feedforward operations.

        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions

        Returns:
            Estimated FLOPs for feedforward computation

        Formula:
            FLOPs_Feedforward = 2 * Batch_Size * Sequence_Length *
                               Hidden_Dimensions * FeedForward_Dimensions
        """
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(sequence_length, "Sequence length")
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")

        result = (
            2
            * batch_size
            * sequence_length
            * hidden_dimensions
            * feedforward_dimensions
        )

        if self._history_callback:
            self._history_callback(
                f"FLOPs_Feedforward(batch_size={batch_size}, sequence_length={sequence_length}, "
                f"hidden_dimensions={hidden_dimensions}, feedforward_dimensions={feedforward_dimensions}) = {result}"
            )
        return result

    def calculate_prefill(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_dimensions: int,
        feedforward_dimensions: int,
        num_layers: int,
    ) -> int:
        """
        Calculate total FLOPs for prefill phase (processing full context).

        Args:
            batch_size: Number of sequences processed in parallel
            sequence_length: Length of input sequences in tokens
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers

        Returns:
            Estimated total FLOPs for prefill computation

        Formula:
            FLOPs_Prefill = Num_Layers * (FLOPs_Attention + FLOPs_Feedforward)
        """
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(sequence_length, "Sequence length")
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")
        validate_positive_integer(num_layers, "Number of layers")

        flops_attention = self.calculate_attention(
            batch_size, sequence_length, hidden_dimensions
        )
        flops_feedforward = self.calculate_feedforward(
            batch_size, sequence_length, hidden_dimensions, feedforward_dimensions
        )

        result = num_layers * (flops_attention + flops_feedforward)

        if self._history_callback:
            self._history_callback(
                f"FLOPs_Prefill(num_layers={num_layers}, "
                f"FLOPs_Attention={flops_attention}, FLOPs_Feedforward={flops_feedforward}) = {result}"
            )
        return result

    def calculate_flops_per_token(
        self,
        batch_size: int,
        hidden_dimensions: int,
        feedforward_dimensions: int,
        num_layers: int,
    ) -> int:
        """
        Calculate FLOPs required for a single token generation.

        Args:
            batch_size: Number of sequences processed in parallel
            hidden_dimensions: Size of hidden dimensions in the model
            feedforward_dimensions: Size of feedforward network dimensions
            num_layers: Number of transformer layers

        Returns:
            Estimated FLOPs for one token computation

        Formula:
            FLOPs_Per_Token = Num_Layers * (FLOPs_Attention_SingleToken + FLOPs_Feedforward_SingleToken)
        """
        validate_positive_integer(batch_size, "Batch size")
        validate_positive_integer(hidden_dimensions, "Hidden dimensions")
        validate_positive_integer(feedforward_dimensions, "Feedforward dimensions")
        validate_positive_integer(num_layers, "Number of layers")

        # For a single new token with kv cache, the sequence length is effectively 1
        flops_attention = self.calculate_attention(batch_size, 1, hidden_dimensions)
        flops_feedforward = self.calculate_feedforward(
            batch_size, 1, hidden_dimensions, feedforward_dimensions
        )

        result = num_layers * (flops_attention + flops_feedforward)

        if self._history_callback:
            self._history_callback(
                f"FLOPs_Per_Token(batch_size={batch_size}, hidden_dimensions={hidden_dimensions}, "
                f"feedforward_dimensions={feedforward_dimensions}, num_layers={num_layers}) = {result}"
            )
        return result
