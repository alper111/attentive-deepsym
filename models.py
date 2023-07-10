import torch
import lightning.pytorch as pl


class DeepSym(pl.LightningModule):
    """DeepSym model from https://arxiv.org/abs/2012.02532"""
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, lr: float, loss_coeff: float = 1.0):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder network.
        decoder : torch.nn.Module
            Decoder network.
        lr : float
            Learning rate.
        loss_coeff : float
            A hyperparameter to increase to speed of convergence when there
            are lots of zero values in the effect prediction (e.g. tile puzzle).
        """
        super(DeepSym, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.module_names = ["encoder", "decoder"]
        self.lr = lr
        self.loss_coeff = loss_coeff

    def encode(self, x: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a state, return its encoding

        Parameters
        ----------
        x : torch.Tensor
            The state tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        h : torch.Tensor
            The code of the given state.
        """
        h = self.encoder(x)
        if eval_mode:
            h = h.round()
        return h

    def concat(self, s: torch.Tensor, a: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a sample, return the concatenation of the encoders'
        output and the action vector.

        Parameters
        ----------
        s : torch.Tensor
            The state tensor.
        a : torch.Tensor
            The action tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        z : torch.Tensor
            The concatenation of the encoder's output and the action vector
            (i.e. the input of the decoder).
        """
        h = self.encode(s, eval_mode)
        z = torch.cat([h, a], dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Given a code, return the effect.

        Parameters
        ----------
        z : torch.Tensor
            The code tensor.

        Returns
        -------
        e : torch.Tensor
            The effect tensor.
        """
        e = self.decoder(z)
        return e

    def forward(self, s: torch.Tensor, a: torch.Tensor, eval_mode=False):
        """
        Given a sample, return the code and the effect.

        Parameters
        ----------
        s : torch.Tensor
            The state tensor.
        a : torch.Tensor
            The action tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        z : torch.Tensor
            The code tensor.
        e : torch.Tensor
            The effect tensor.
        """
        z = self.concat(s, a, eval_mode)
        e = self.decode(z)
        return z, e

    def training_step(self, batch, _):
        s, a, e, _, _ = batch
        _, e_pred = self.forward(s, a)
        loss = torch.nn.functional.mse_loss(e_pred, e) * self.loss_coeff
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        s, a, e, _, _ = batch
        _, e_pred = self.forward(s, a, eval_mode=True)
        loss = torch.nn.functional.mse_loss(e_pred, e) * self.loss_coeff
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, _, _ = batch
        _, e_pred = self.forward(s, a, eval_mode=True)
        loss = torch.nn.functional.mse_loss(e_pred, e) * self.loss_coeff
        return loss

    def predict_step(self, batch, _):
        s, a, _, _, sn = batch
        z, e = self.forward(s, a, eval_mode=True)
        zn = self.encode(sn, eval_mode=True)
        return {"z": z, "e": e, "zn": zn}

    def configure_optimizers(self):
        params = []
        for name in self.module_names:
            module = getattr(self, name)
            for p in module.parameters():
                params.append(p)
        optimizer = torch.optim.Adam(lr=self.lr, params=params)
        return optimizer


class AttentiveDeepSym(DeepSym):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, post_encoder: torch.nn.Module,
                 attention: torch.nn.Module, pre_attention: torch.nn.Module, lr: float, loss_coeff: float = 1):
        super(AttentiveDeepSym, self).__init__(encoder, decoder, lr, loss_coeff)
        self.post_encoder = post_encoder
        self.attention = attention
        self.pre_attention = pre_attention
        self.module_names += ["post_encoder", "attention", "pre_attention"]

    def encode(self, x: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a state, return its encoding

        Parameters
        ----------
        x : torch.Tensor
            The state tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        h : torch.Tensor
            The code of the given state.
        """
        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        h = self.encoder(x)
        h = h.reshape(n_sample, n_seg, -1)
        if eval_mode:
            h = h.round()
        return h

    def attn_weights(self, x: torch.Tensor, pad_mask: torch.Tensor, eval_mode=False) -> torch.Tensor:
        # assume that x is not an image for the moment..
        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        x = self.pre_attention(x)
        x = x.reshape(n_sample, n_seg, -1)
        attn_weights = self.attention(x, src_key_mask=pad_mask)
        if eval_mode:
            attn_weights = attn_weights.round()
        return attn_weights

    def aggregate(self, z: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        n_batch, n_seg, n_dim = z.shape
        post_h = self.post_encoder(z.reshape(-1, n_dim)).reshape(n_batch, n_seg, -1).unsqueeze(1)
        att_out = attn_weights @ post_h  # (n_batch, n_head, n_seg, n_dim)
        att_out = att_out.permute(0, 2, 1, 3).reshape(n_batch, n_seg, -1)  # (n_batch, n_seg, n_head*n_dim)
        return att_out

    def decode(self, z: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        n_sample, n_seg, z_dim = z.shape
        z = z.reshape(-1, z_dim)
        e = self.decoder(z)
        e = e.reshape(n_sample, n_seg, -1)
        pad_mask = pad_mask.reshape(n_sample, n_seg, 1)
        # turn off computation for padded parts
        e_masked = e * pad_mask
        return e_masked

    def forward(self, s: torch.Tensor, a: torch.Tensor, pad_mask: torch.Tensor, eval_mode=False):
        z = self.concat(s, a, eval_mode)
        attn_weights = self.attn_weights(s, pad_mask, eval_mode)
        z_att = self.aggregate(z, attn_weights)
        e = self.decode(z_att, pad_mask)
        return z, attn_weights, e

    def training_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask, eval_mode=True)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask, eval_mode=True)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        return loss

    def predict_step(self, batch, _):
        s, a, _, pad_mask, sn = batch
        z, r, e = self.forward(s, a, pad_mask, eval_mode=True)
        zn = self.encode(sn, eval_mode=True)
        rn = self.attn_weights(sn, pad_mask, eval_mode=True)
        return {"z": z, "r": r, "e": e, "zn": zn, "rn": rn}
