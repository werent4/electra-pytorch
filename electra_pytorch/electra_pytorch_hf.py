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
    AutoTokenizer
)
from typing import Union
from electra_pytorch.electra_pytorch import gumbel_sample

class Trainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=True):  
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        generator_save_path = f"{output_dir}/generator"
        discriminator_save_path = f"{output_dir}/discriminator"
        os.makedirs(generator_save_path, exist_ok= True)
        os.makedirs(discriminator_save_path, exist_ok= True)
        self.model.generator.save_pretrained(generator_save_path)
        self.model.discriminator.save_pretrained(discriminator_save_path)

        print(f"Custom model saved in {output_dir}")

class ElectraHuggingFace(nn.Module):
    def __init__(
        self,
        generator: PreTrainedModel,
        generator_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast], 
        discriminator: PreTrainedModel,
        *,
        mask_ignore_token_ids = [],
        disc_weight = 50., 
        gen_weight = 1., 
        temperature = 1.
        ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        if self.generator.config.vocab_size != self.discriminator.config.vocab_size:
            raise ValueError("generator and discriminator models have different vocab size") 

        # token ids
        self.pad_token_id = generator_tokenizer.pad_token_id
        self.mask_token_id = generator_tokenizer.mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, self.pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight 
        self.gen_weight = gen_weight 

    def forward(self, input= None, input_ids=None, attention_mask=None, labels=None,**kwargs):
        # Extract inputs from the batch
        if input is not None:
            input_ids = input["input_ids"]
            attention_mask = input.get("attention_mask", None)
            labels = input.get("labels", None)

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
        disc_labels = (input_ids != disc_input).float().detach() # Метки для дискриминатора

        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input_ids != self.pad_token_id, as_tuple=True)
        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(input_ids=disc_input, attention_mask=attention_mask, **kwargs).logits # discriminator logits
        disc_logits = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits( # discriminator loss
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        # return weighted sum of losses
        loss = self.gen_weight * mlm_loss + self.disc_weight * disc_loss

        if torch.isnan(loss):
            logging.error("NaN detected in total loss.")

        return {
            "loss": loss, 
            "logits": logits,
            "disc_logits": disc_logits, 
            }
    @classmethod
    def from_pretrained(cls, output_dir, mask_ignore_token_ids=[], disc_weight=50., gen_weight=1., temperature=1.):
        """
        Кастомный метод `from_pretrained`, который загружает генератор, дискриминатор и токенизатор из директории.
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