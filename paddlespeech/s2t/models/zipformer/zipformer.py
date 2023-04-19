import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import random

class Zipformer(nn.Layer):
    """
    Args:
        num_features (int): Number of input features
        d_model: (int,int): embedding dimension of 2 encoder stacks
        attention_dim: (int,int): attention dimension of 2 encoder stacks
        nhead (int, int): number of heads
        dim_feedforward (int, int): feedforward dimension in 2 encoder stacks
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
        cnn_module_kernel (int): Kernel size of convolution module
        vgg_frontend (bool): whether to use vgg frontend.
        warmup_batches (float): number of batches to warm up over
    """

    def __init__(
        self,
        num_features: int,
        output_downsampling_factor: int = 2,
        encoder_dims: Tuple[int] = (384, 384),
        attention_dim: Tuple[int] = (256, 256),
        encoder_unmasked_dims: Tuple[int] = (256, 256),
        zipformer_downsampling_factors: Tuple[int] = (2, 4),
        nhead: Tuple[int] = (8, 8),
        feedforward_dim: Tuple[int] = (1536, 2048),
        num_encoder_layers: Tuple[int] = (12, 12),
        dropout: float = 0.1,
        cnn_module_kernels: Tuple[int] = (31, 31),
        pos_dim: int = 4,
        warmup_batches: float = 4000.0,
    ) -> None:
        super(Zipformer, self).__init__()

        self.num_features = num_features
        assert 0 < encoder_dims[0] <= encoder_dims[1]
        self.encoder_dims = encoder_dims
        self.encoder_unmasked_dims = encoder_unmasked_dims
        self.zipformer_downsampling_factors = zipformer_downsampling_factors
        self.output_downsampling_factor = output_downsampling_factor

        # will be written to, see set_batch_count()
        self.batch_count = 0
        self.warmup_end = warmup_batches

        for u, d in zip(encoder_unmasked_dims, encoder_dims):
            assert u <= d, (u, d)

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T - 7)//2, encoder_dims).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T - 7)//2
        #   (2) embedding: num_features -> encoder_dims
        self.encoder_embed = Conv2dSubsampling(
            num_features, encoder_dims[0], dropout=dropout
        )

        # each one will be ZipformerEncoder or DownsampledZipformerEncoder
        encoders = []

        num_encoders = len(encoder_dims)
        for i in range(num_encoders):
            encoder_layer = ZipformerEncoderLayer(
                encoder_dims[i],
                attention_dim[i],
                nhead[i],
                feedforward_dim[i],
                dropout,
                cnn_module_kernels[i],
                pos_dim,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = ZipformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
            )

            if zipformer_downsampling_factors[i] != 1:
                encoder = DownsampledZipformerEncoder(
                    encoder,
                    input_dim=encoder_dims[i - 1] if i > 0 else encoder_dims[0],
                    output_dim=encoder_dims[i],
                    downsample=zipformer_downsampling_factors[i],
                )
            encoders.append(encoder)
        self.encoders = nn.LayerList(encoders)

        # initializes self.skip_layers and self.skip_modules
        self._init_skip_modules()

        self.downsample_output = AttentionDownsample(
            encoder_dims[-1], encoder_dims[-1], downsample=output_downsampling_factor
        )

    def _get_layer_skip_dropout_prob(self):
        if not self.training:
            return 0.0
        batch_count = self.batch_count
        min_dropout_prob = 0.025

        if batch_count > self.warmup_end:
            return min_dropout_prob
        else:
            return 0.5 - (batch_count / self.warmup_end) * (0.5 - min_dropout_prob)

    def _init_skip_modules(self):
        """
        If self.zipformer_downampling_factors = (1, 2, 4, 8, 4, 2), then at the input of layer
        indexed 4 (in zero indexing), with has subsapling_factor=4, we combine the output of
        layers 2 and 3; and at the input of layer indexed 5, which which has subsampling_factor=2,
        we combine the outputs of layers 1 and 5.
        """
        skip_layers = []
        skip_modules = []
        z = self.zipformer_downsampling_factors
        for i in range(len(z)):
            if i <= 1 or z[i - 1] <= z[i]:
                skip_layers.append(None)
                skip_modules.append(SimpleCombinerIdentity())
            else:
                # TEMP
                for j in range(i - 2, -1, -1):
                    if z[j] <= z[i] or j == 0:
                        # TEMP logging statement.
                        logging.info(
                            f"At encoder stack {i}, which has downsampling_factor={z[i]}, we will "
                            f"combine the outputs of layers {j} and {i-1}, with downsampling_factors={z[j]} and {z[i-1]}."
                        )
                        skip_layers.append(j)
                        skip_modules.append(
                            SimpleCombiner(
                                self.encoder_dims[j],
                                self.encoder_dims[i - 1],
                                min_weight=(0.0, 0.25),
                            )
                        )
                        break
        self.skip_layers = skip_layers
        self.skip_modules = nn.LayerList(skip_modules)

    def get_feature_masks(self, x: paddle.Tensor) -> List[float]:
        # Note: The actual return type is Union[List[float], List[Tensor]],
        # but to make torch.jit.script() work, we use List[float]
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all encoder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoder dim.
        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_downsampling_factors times.
        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (num_frames, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dims)
        if not self.training:
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dims[0] == _encoder_dims0, (
            self.encoder_dims,
            _encoder_dims0,
        )

        max_downsampling_factor = max(self.zipformer_downsampling_factors)

        num_frames_max = num_frames0 + max_downsampling_factor - 1

        feature_mask_dropout_prob = 0.15

        # frame_mask_max shape: (num_frames_max, batch_size, 1)
        frame_mask_max = (
            paddle.rand([num_frames_max, batch_size, 1])
            > feature_mask_dropout_prob
        ).astype(x.dtype)

        feature_masks = []
        for i in range(num_encoders):
            ds = self.zipformer_downsampling_factors[i]
            upsample_factor = max_downsampling_factor // ds

            frame_mask = (
                frame_mask_max.unsqueeze(1)
                .expand([num_frames_max, upsample_factor, batch_size, 1])
                .reshape([num_frames_max * upsample_factor, batch_size, 1])
            )
            num_frames = (num_frames0 + ds - 1) // ds
            frame_mask = frame_mask[:num_frames]
            feature_mask = paddle.ones(
                [num_frames,
                batch_size,
                self.encoder_dims[i]],
                dtype=x.dtype)
            u = self.encoder_unmasked_dims[i]
            feature_mask[:, :, u:] *= frame_mask
            feature_masks.append(feature_mask)

        return feature_masks

    def forward(
        self,
        x: paddle.Tensor,
        x_lens: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, encoder_dims[-1])
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        lengths = (x_lens - 7) >> 1
        assert x.size(0) == lengths.max().item(), (x.shape, lengths, lengths.max())
        mask = make_pad_mask(lengths)

        outputs = []
        feature_masks = self.get_feature_masks(x)

        for i, (module, skip_module) in enumerate(
            zip(self.encoders, self.skip_modules)
        ):
            ds = self.zipformer_downsampling_factors[i]
            k = self.skip_layers[i]
            if isinstance(k, int):
                layer_skip_dropout_prob = self._get_layer_skip_dropout_prob()
                if (not self.training) or random.random() > layer_skip_dropout_prob:
                    x = skip_module(outputs[k], x)
            x = module(
                x,
                feature_mask=feature_masks[i],
                src_key_padding_mask=None if mask is None else mask[..., ::ds],
            )
            outputs.append(x)

        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        lengths = (lengths + 1) >> 1

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths


class ZipformerEncoderLayer(nn.Layer):
    """
    ZipformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.
    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
        >>> src = paddle.rand([10, 32, 512])
        >>> pos_emb = paddle.rand([32, 19, 512])
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        attention_dim: int,
        nhead: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        cnn_module_kernel: int = 31,
        pos_dim: int = 4,
    ) -> None:
        super(ZipformerEncoderLayer, self).__init__()

        self.d_model = d_model

        # will be written to, see set_batch_count()
        self.batch_count = 0

        self.self_attn = RelPositionMultiheadAttention(
            d_model,
            attention_dim,
            nhead,
            pos_dim,
            dropout=0.0,
        )

        self.pooling = PoolingModule(d_model)

        self.feed_forward1 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.feed_forward2 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(d_model, feedforward_dim, dropout)

        self.conv_module1 = ConvolutionModule(d_model, cnn_module_kernel)

        self.conv_module2 = ConvolutionModule(d_model, cnn_module_kernel)

        self.norm_final = BasicNorm(d_model)

        self.bypass_scale = paddle.create_parameter(shape=[1], dtype='float32', default_initializer=nn.initializer.Assign(paddle.to_tensor(0.5)))

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            d_model,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            max_abs=6.0,
        )
        self.whiten = Whiten(
            num_groups=1, whitening_limit=5.0, prob=(0.025, 0.25), grad_scale=0.01
        )

    def get_bypass_scale(self):
        if not self.training:
            return self.bypass_scale
        if random.random() < 0.1:
            # ensure we get grads if self.bypass_scale becomes out of range
            return self.bypass_scale
        # hardcode warmup period for bypass scale
        warmup_period = 20000.0
        initial_clamp_min = 0.75
        final_clamp_min = 0.25
        if self.batch_count > warmup_period:
            clamp_min = final_clamp_min
        else:
            clamp_min = initial_clamp_min - (self.batch_count / warmup_period) * (
                initial_clamp_min - final_clamp_min
            )
        return self.bypass_scale.clamp(min=clamp_min, max=1.0)

    def get_dynamic_dropout_rate(self):
        # return dropout rate for the dynamic modules (self_attn, pooling, convolution); this
        # starts at 0.2 and rapidly decreases to 0.  Its purpose is to keep the training stable
        # at the beginning, by making the network focus on the feedforward modules.
        if not self.training:
            return 0.0
        warmup_period = 2000.0
        initial_dropout_rate = 0.2
        final_dropout_rate = 0.0
        if self.batch_count > warmup_period:
            return final_dropout_rate
        else:
            return initial_dropout_rate - (
                initial_dropout_rate * final_dropout_rate
            ) * (self.batch_count / warmup_period)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        batch_split: if not None, this layer will only be applied to
        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        src_orig = src

        # macaron style feed forward module
        src = src + self.feed_forward1(src)

        # dropout rate for submodules that interact with time.
        dynamic_dropout = self.get_dynamic_dropout_rate()

        # pooling module
       
        if random.random() >= dynamic_dropout:
            src = src + self.pooling(src, key_padding_mask=src_key_padding_mask)

        use_self_attn = random.random() >= dynamic_dropout
        if use_self_attn:
            src_att, attn_weights = self.self_attn(
                src,
                pos_emb=pos_emb,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + src_att

        if random.random() >= dynamic_dropout:
            src = src + self.conv_module1(
                src, src_key_padding_mask=src_key_padding_mask
            )

        src = src + self.feed_forward2(src)
        if use_self_attn:
            src = src + self.self_attn.forward2(src, attn_weights)

        if random.random() >= dynamic_dropout:
            src = src + self.conv_module2(
                src, src_key_padding_mask=src_key_padding_mask
            )

        src = src + self.feed_forward3(src)

        src = self.norm_final(self.balancer(src))

        delta = src - src_orig

        src = src_orig + delta * self.get_bypass_scale()

        return self.whiten(src)


class ZipformerEncoder(nn.Layer):
    r"""ZipformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the ZipformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(d_model=512, nhead=8)
        >>> zipformer_encoder = ZipformerEncoder(encoder_layer, num_layers=6)
        >>> src = paddle.rand([10, 32, 512])
        >>> out = zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
    ) -> None:
        super().__init__()
        # will be written to, see set_batch_count() Note: in inference time this
        # may be zero but should be treated as large, we can check if
        # self.training is true.
        self.batch_count = 0
        self.warmup_begin = warmup_begin
        self.warmup_end = warmup_end
        # module_seed is for when we need a random number that is unique to the module but
        # shared across jobs.   It's used to randomly select how many layers to drop,
        # so that we can keep this consistent across worker tasks (for efficiency).
        self.module_seed = paddle.randint(0, 1000, (1)).item()

        self.encoder_pos = RelPositionalEncoding(encoder_layer.d_model, dropout)

        self.layers = nn.LayerList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin
        for i in range(num_layers):
            self.layers[i].warmup_begin = cur_begin
            cur_begin += delta
            self.layers[i].warmup_end = cur_begin

    def get_layers_to_drop(self, rnd_seed: int):
        ans = set()
        if not self.training:
            return ans

        batch_count = self.batch_count
        num_layers = len(self.layers)

        def get_layerdrop_prob(layer: int) -> float:
            layer_warmup_begin = self.layers[layer].warmup_begin
            layer_warmup_end = self.layers[layer].warmup_end

            initial_layerdrop_prob = 0.5
            final_layerdrop_prob = 0.05

            if batch_count == 0:
                # As a special case, if batch_count == 0, return 0 (drop no
                # layers).  This is rather ugly, I'm afraid; it is intended to
                # enable our scan_pessimistic_batches_for_oom() code to work correctly
                # so if we are going to get OOM it will happen early.
                # also search for 'batch_count' with quotes in this file to see
                # how we initialize the warmup count to a random number between
                # 0 and 10.
                return 0.0
            elif batch_count < layer_warmup_begin:
                return initial_layerdrop_prob
            elif batch_count > layer_warmup_end:
                return final_layerdrop_prob
            else:
                # linearly interpolate
                t = (batch_count - layer_warmup_begin) / layer_warmup_end
                assert 0.0 <= t < 1.001, t
                return initial_layerdrop_prob + t * (
                    final_layerdrop_prob - initial_layerdrop_prob
                )

        shared_rng = random.Random(batch_count + self.module_seed)
        independent_rng = random.Random(rnd_seed)

        layerdrop_probs = [get_layerdrop_prob(i) for i in range(num_layers)]
        tot = sum(layerdrop_probs)
        # Instead of drawing the samples independently, we first randomly decide
        # how many layers to drop out, using the same random number generator between
        # jobs so that all jobs drop out the same number (this is for speed).
        # Then we use an approximate approach to drop out the individual layers
        # with their specified probs while reaching this exact target.
        num_to_drop = int(tot) + int(shared_rng.random() < (tot - int(tot)))

        layers = list(range(num_layers))
        independent_rng.shuffle(layers)

        # go through the shuffled layers until we get the required number of samples.
        if num_to_drop > 0:
            for layer in itertools.cycle(layers):
                if independent_rng.random() < layerdrop_probs[layer]:
                    ans.add(layer)
                if len(ans) == num_to_drop:
                    break
        if shared_rng.random() < 0.005 or __name__ == "__main__":
            logging.info(
                f"warmup_begin={self.warmup_begin:.1f}, warmup_end={self.warmup_end:.1f}, "
                f"batch_count={batch_count:.1f}, num_to_drop={num_to_drop}, layers_to_drop={ans}"
            )
        return ans

    def forward(
        self,
        src: Tensor,
        # Note: The type of feature_mask should be Union[float, Tensor],
        # but to make torch.jit.script() work, we use `float` here
        feature_mask: float = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        Returns: (x, x_no_combine), both of shape (S, N, E)
        """
        pos_emb = self.encoder_pos(src)
        output = src

        rnd_seed = src.numel() + random.randint(0, 1000)
        layers_to_drop = self.get_layers_to_drop(rnd_seed)

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
            if i in layers_to_drop:
                continue
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            output = output * feature_mask

        return output

class DownsampledZipformerEncoder(nn.Layer):
    r"""
    DownsampledZipformerEncoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """

    def __init__(
        self, encoder: nn.Module, input_dim: int, output_dim: int, downsample: int
    ):
        super(DownsampledZipformerEncoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = AttentionDownsample(input_dim, output_dim, downsample)
        self.encoder = encoder
        self.upsample = SimpleUpsample(output_dim, downsample)
        self.out_combiner = SimpleCombiner(
            input_dim, output_dim, min_weight=(0.0, 0.25)
        )

    def forward(
        self,
        src: Tensor,
        # Note: the type of feature_mask should be Unino[float, Tensor],
        # but to make torch.jit.script() happ, we use float here
        feature_mask: float = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.
        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            mask: the mask for the src sequence (optional).  CAUTION: we need to downsample
                  this, if we are to support it.  Won't work correctly yet.
            src_key_padding_mask: the mask for the src keys per batch (optional).  Should
                  be downsampled already.
        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if mask is not None:
            mask = mask[::ds, ::ds]

        src = self.encoder(
            src,
            feature_mask=feature_mask,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)

class AttentionDownsample(nn.Layer):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: int):
        """
        Require out_channels > in_channels.
        """
        super(AttentionDownsample, self).__init__()
        self.query = paddle.create_parameter(shape=[in_channels], dtype='float32', default_initializer=nn.initializer.Assign(paddle.randn([in_channels]) * (in_channels**-0.5)))

        # fill in the extra dimensions with a projection of the input
        if out_channels > in_channels:
            self.extra_proj = Linear(
                in_channels * downsample, out_channels - in_channels, bias=False
            )
        else:
            self.extra_proj = None
        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample, could be 0 for onnx-export-compatibility
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand([pad, src.shape[1], src.shape[2]])
        src = paddle.concat((src, src_extra), axis=0)
        assert src.shape[0] == d_seq_len * ds, (src.shape[0], d_seq_len, ds)

        src = src.reshape([d_seq_len, ds, batch_size, in_channels])
        scores = (src * self.query).sum(axis=-1, keepdim=True)

        scores = penalize_abs_values_gt(scores, limit=10.0, penalty=1.0e-04)

        weights = F.softmax(scores, axis=1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(axis=1)
        src = src.transpose([0, 2, 1, 3]).reshape([d_seq_len, batch_size, ds * in_channels])

        if self.extra_proj is not None:
            ans2 = self.extra_proj(src)
            ans = paddle.concat((ans, ans2), axis=2)
        return ans

class SimpleUpsample(nn.Layer):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.bias = paddle.create_parameter(shape=[upsample, num_channels], dtype='float32', default_initializer=nn.initializer.Assign(paddle.randn([upsample, num_channels]) * 0.01))

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.bias.shape[0]
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src + self.bias.unsqueeze(1)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


class SimpleCombinerIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        return src1


class SimpleCombiner(torch.nn.Module):
    """
    A very simple way of combining 2 vectors of 2 different dims, via a
    learned weighted combination in the shared part of the dim.
    Args:
         dim1: the dimension of the first input, e.g. 256
         dim2: the dimension of the second input, e.g. 384.
    The output will have the same dimension as dim2.
    """

    def __init__(self, dim1: int, dim2: int, min_weight: Tuple[float] = (0.0, 0.0)):
        super(SimpleCombiner, self).__init__()
        assert dim2 >= dim1, (dim2, dim1)
        self.weight1 = nn.Parameter(torch.zeros(()))
        self.min_weight = min_weight

    def forward(self, src1: Tensor, src2: Tensor) -> Tensor:
        """
        src1: (*, dim1)
        src2: (*, dim2)
        Returns: a tensor of shape (*, dim2)
        """
        assert src1.shape[:-1] == src2.shape[:-1], (src1.shape, src2.shape)

        weight1 = self.weight1
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            if (
                self.training
                and random.random() < 0.25
                and self.min_weight != (0.0, 0.0)
            ):
                weight1 = weight1.clamp(
                    min=self.min_weight[0], max=1.0 - self.min_weight[1]
                )

        src1 = src1 * weight1
        src2 = src2 * (1.0 - weight1)

        src1_dim = src1.shape[-1]
        src2_dim = src2.shape[-1]
        if src1_dim != src2_dim:
            if src1_dim < src2_dim:
                src1 = torch.nn.functional.pad(src1, (0, src2_dim - src1_dim))
            else:
                src1 = src1[:src2_dim]

        return src1 + src2