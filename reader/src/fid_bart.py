"""Fusion-in-Decoder BART for document-separated encoder inputs."""

from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class FiDBart(BartForConditionalGeneration):
    """Encode B x N x L passages independently and fuse before decoding."""

    def _encode_passages(self, input_ids, attention_mask):
        batch_size, n_passages, passage_length = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * n_passages, passage_length)
        flat_mask = attention_mask.reshape(batch_size * n_passages, passage_length)
        encoded = self.model.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            return_dict=True,
        )
        hidden = encoded.last_hidden_state.reshape(
            batch_size, n_passages * passage_length, -1
        )
        return BaseModelOutput(last_hidden_state=hidden), attention_mask.reshape(
            batch_size, n_passages * passage_length
        )

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None,
                **kwargs):
        if encoder_outputs is None and input_ids is not None and input_ids.dim() == 3:
            encoder_outputs, attention_mask = self._encode_passages(
                input_ids, attention_mask
            )
            input_ids = None
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
