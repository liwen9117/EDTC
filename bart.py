import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
import os

class BART(nn.Module):
    def __init__(self, settings, device):
        super().__init__()
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(settings["data"]["tokenizer"])

        self.num_beams = settings['generation']['num_beams']

        self.bart_lm = BartForConditionalGeneration.from_pretrained('./checkpoint/bart-base')
        # print(self.bart_lm.config)

    # 右移input_id
    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    # 定义贪心算法来生成文本
    def generate_greedy(self,
        inputs_embeds=None
    ):

        encoder_outputs = self.bart_lm.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True
        )

        max_len = self.bart_lm.config.max_length
        cur_len = 0
        input_ids = torch.zeros((inputs_embeds.size(0), 1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
         
        # print(input)
        
        outputs = self.bart_lm(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=input_ids,
            decoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True
        )

        _next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
        _past = outputs['past_key_values']
        _encoder_last_hidden_state = outputs['encoder_last_hidden_state']
        input_ids = torch.cat([input_ids, _next_token.unsqueeze(-1)], dim=-1)

        # Override with bos token
        input_ids[:, 1] = self.bart_lm.config.bos_token_id
        cur_len += 1

        while cur_len < max_len-1:
            model_inputs = self.bart_lm.prepare_inputs_for_generation(
                input_ids, 
                past=_past,
                attention_mask=None,
                encoder_outputs=[
                    encoder_outputs['last_hidden_state']]
            )
            outputs = self.bart_lm(**model_inputs)
            _next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
            _past = outputs['past_key_values']
            _encoder_last_hidden_state = outputs['encoder_last_hidden_state']
            input_ids = torch.cat([input_ids, _next_token.unsqueeze(-1)], dim=-1)
            cur_len += 1
        
        return input_ids

    # 定义beam search算法来生成文本
    def generate_beam(self,
        inputs_embeds=None,
    ):
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True
        )
        
        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        # Beam decoding
        outputs = self.bart_lm.generate(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            head_mask=None,
            decoder_head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            num_beams=self.num_beams,
        )
    
        return outputs
    
    # 定义nucleus sampling算法来生成文本
    def generate_sampling(self,
        inputs_embeds=None,
    ):
        
        # Encoder pass
        encoder_outputs = self.bart_lm.model.encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True
        )
        
        input_ids = torch.zeros((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs['last_hidden_state'].size(0),1)).long().to(self.device)
        # Beam decoding
        outputs = self.bart_lm.generate(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            head_mask=None,
            decoder_head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            num_beams=self.num_beams,
            do_sample=True,
            temperature=0.5,
            top_p=0.95
        )
    
        return outputs
    
    def forward(self,
        decoder_attention_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # Encoder pass  (batch_size, sequence_length, hidden_size)   
        encoder_outputs = self.bart_lm.model.encoder(
            inputs_embeds=inputs_embeds,
            return_dict=True)['last_hidden_state']

        # 1 -> -100
        decoder_targets = labels.masked_fill(           
            labels == self.tokenizer.pad_token_id, -100
        )

        # 在前面加个2
        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.bart_lm.config.pad_token_id, self.bart_lm.config.decoder_start_token_id
        )

        # Decoder-only pass
        outputs = self.bart_lm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=None,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=(encoder_outputs,),
            labels=None,
            return_dict=True,
        )

        logits = outputs["logits"]
        decoder_last_hidden_state = outputs["decoder_hidden_states"]

        return logits, decoder_last_hidden_state

if __name__ == "__main__":
    print(os.getcwd())


