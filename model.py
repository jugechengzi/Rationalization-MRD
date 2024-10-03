import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)






class Sp_norm_model_causal(nn.Module):
    def __init__(self, args):
        super(Sp_norm_model_causal, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)


        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

        #4.21加的，单独的分类器，用来训练好generator之后单独训练分类器
        self.post_cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)

        self.post_cls_fc = nn.Linear(args.hidden_dim, args.num_class)



    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def post_forward(self, inputs, masks):
        """
        训练好generator之后，用rationale单独训练post_classifier
        :param inputs:
        :param masks:
        :return:
        """
        self.generator.eval()
        with torch.no_grad():
            masks_ = masks.unsqueeze(-1)

            ########## Genetator ##########
            embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
            # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
            # if self.lay:
            #     gen_output = self.layernorm1(gen_output)
            # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
            gen_logits = self.generator(embedding)
            ########## Sample ##########
            z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
            ########## Classifier ##########
            cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.post_cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.post_cls_fc(self.dropout(cls_outputs))
        return z, cls_logits


    def train_one_step(self, inputs, masks):    #input x directly to predictor
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def get_rationale(self, inputs, masks):
        """
        获取generator生成的rationale (binary)
        :param inputs:
        :param masks:
        :return: z=[batchsize,seq_len,2]
        """

        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z
    def pred_forward_logit(self, inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def adversarial_pred_logit(self,inputs,masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        #采用为0的部分来预测
        cls_embedding = embedding * (z[:, :, 0].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)


        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits


    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log


class Sp_norm_rnp(nn.Module):
    """
    20240423新版rnp
    """
    def __init__(self, args):
        super(Sp_norm_rnp, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)


        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)




    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def post_forward(self, inputs, masks):
        """
        训练好generator之后，用rationale单独训练post_classifier
        :param inputs:
        :param masks:
        :return:
        """
        self.generator.eval()
        with torch.no_grad():
            masks_ = masks.unsqueeze(-1)

            ########## Genetator ##########
            embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
            # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
            # if self.lay:
            #     gen_output = self.layernorm1(gen_output)
            # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
            gen_logits = self.generator(embedding)
            ########## Sample ##########
            z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
            ########## Classifier ##########
            cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.post_cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.post_cls_fc(self.dropout(cls_outputs))
        return z, cls_logits


    def train_one_step(self, inputs, masks):    #input x directly to predictor
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def get_rationale(self, inputs, masks):
        """
        获取generator生成的rationale (binary)
        :param inputs:
        :param masks:
        :return: z=[batchsize,seq_len,2]
        """

        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = self.generator(embedding)
        ########## Sample ##########
        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z
    def pred_forward_logit(self, inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def adversarial_pred_logit(self,inputs,masks,z):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)

        ########## Classifier ##########
        #采用为0的部分来预测
        cls_embedding = embedding * (z[:, :, 0].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)


        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits


    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log










