import math
from functools import reduce
from collections import namedtuple
import torch, logging, os
from torch import nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Trainer,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    ElectraForMaskedLM,
    ElectraForPreTraining
)
from transformers.utils import ModelOutput
from typing import Union, Optional
from dataclasses import dataclass
from electra_pytorch.electra_pytorch import gumbel_sample
from .loss_functions import focal_loss_with_logits

class Trainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=True):  
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        generator_save_path = f"{output_dir}/generator"
        discriminator_save_path = f"{output_dir}/discriminator"
        os.makedirs(generator_save_path, exist_ok= True)
        os.makedirs(discriminator_save_path, exist_ok= True)
        self.model.generator.save_pretrained(generator_save_path)
        self.model.discriminator.save_pretrained(discriminator_save_path)

@dataclass
class ElectraOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    disc_logits: Optional[torch.FloatTensor] = None

class ElectraHuggingFace(nn.Module):
    def __init__(
        self,
        generator: PreTrainedModel,
        generator_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast], 
        discriminator: PreTrainedModel,
        *,
        device: torch.device = torch.device('cpu'),
        mask_ignore_token_ids = [],
        disc_weight = 50., 
        gen_weight = 1., 
        temperature = 1.
        ):
        super().__init__()

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device

        if self.generator.config.vocab_size != self.discriminator.config.vocab_size:
            raise ValueError("generator and discriminator models have different vocab size") 
        
        self.tie_weights()

        # token ids
        self.pad_token_id = generator_tokenizer.pad_token_id
        self.mask_token_id = generator_tokenizer.mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, self.pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight 
        self.gen_weight = gen_weight 

    def forward(self, input= None, input_ids=None, attention_mask=None, labels=None, original_input_ids=None, **kwargs):
        if input_ids is None or attention_mask is None or labels is None:
            raise ValueError("`input_ids`, `attention_mask`, and `labels` cannot be None.")
        
        # Extract inputs from the batch
        if input is not None:
            input_ids = input["input_ids"].to(self.device)
            attention_mask = input.get("attention_mask", None).to(self.device) 
            labels = input.get("labels", None).to(self.device)
            original_input_ids = input.get("original_input_ids", None).to(self.device)

        input_ids = input_ids.to(self.device) 
        original_input_ids = original_input_ids.to(self.device) 
        attention_mask = attention_mask.to(self.device) 
        labels = labels.to(self.device) 

        logits = self.generator(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits # Generator logits

        mlm_loss = F.cross_entropy( # generator loss
            logits.transpose(1, 2),
            labels,
            ignore_index = -100
        )

        # Sample generator predictions for discriminator input
        with torch.no_grad():
            sampled = gumbel_sample(logits, temperature = self.temperature) # Sampling using generator logits
            # scatter the sampled values back to the input
            disc_input = input_ids.clone()
            disc_input[labels != -100] = sampled[labels != -100] # Preparing the input for the discriminator

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (original_input_ids != disc_input).float().detach() # Метки для дискриминатора

        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input_ids != self.pad_token_id, as_tuple=True)
        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(input_ids=disc_input, attention_mask=attention_mask, **kwargs).logits # discriminator logits
        disc_logits = disc_logits.reshape_as(disc_labels)

        disc_loss = focal_loss_with_logits(
            inputs=disc_logits[non_padded_indices],
            targets=disc_labels[non_padded_indices],
            reduction="mean", 
        )

        # return weighted sum of losses
        loss = self.gen_weight * mlm_loss + self.disc_weight * disc_loss

        if torch.isnan(loss):
            logging.error("NaN detected in total loss.")

        return ElectraOutput(
            loss=loss, 
            logits=logits,
            disc_logits=disc_logits
        )
    
    def tie_weights(self):
        self.generator.base_model.embeddings.word_embeddings = self.discriminator.base_model.embeddings.word_embeddings
        self.generator.base_model.embeddings.position_embeddings = self.discriminator.base_model.embeddings.position_embeddings
        if hasattr(self.generator.base_model.embeddings, 'token_type_embeddings') and hasattr(self.discriminator.base_model.embeddings, 'token_type_embeddings'):
            self.generator.base_model.embeddings.token_type_embeddings = self.discriminator.base_model.embeddings.token_type_embeddings

    @classmethod
    def from_pretrained(cls, output_dir, mask_ignore_token_ids=[], disc_weight=50., gen_weight=1., temperature=1.):
        """
        Custom method `from_pretrained` which loads the generator, discriminator and tokenizer from a directory.
        """
        # Loading the generator and discriminator from the corresponding paths
        generator = AutoModelForMaskedLM.from_pretrained(os.path.join(output_dir, "generator"))
        discriminator = AutoModelForTokenClassification.from_pretrained(
            os.path.join(output_dir, "discriminator"),
            num_labels=1
        )

        # Load tokenizer
        generator_tokenizer = AutoTokenizer.from_pretrained(output_dir)

        return cls(
            generator=generator,
            generator_tokenizer=generator_tokenizer,
            discriminator=discriminator,
            mask_ignore_token_ids=mask_ignore_token_ids,
            disc_weight=disc_weight,
            gen_weight=gen_weight,
            temperature=temperature
        )