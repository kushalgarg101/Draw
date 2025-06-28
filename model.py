import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, img_width, img_height, img_channels, N, dec_hidden_dim):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.N = N

        self.fc_attn_params_read = nn.Linear(dec_hidden_dim, 5)
        self.fc_attn_params_write = nn.Linear(dec_hidden_dim, 5)
        self.fc_write_patch = nn.Linear(dec_hidden_dim, N * N * img_channels)


    def _get_filter_means(self, g_x, g_y, delta):
        i_offsets = torch.arange(self.N, dtype=g_x.dtype, device=g_x.device) - self.N / 2.0 + 0.5

        mu_X = g_x + i_offsets.unsqueeze(0) * delta
        mu_Y = g_y + i_offsets.unsqueeze(0) * delta

        return mu_X, mu_Y

    def _get_filterbanks(self, mu_x, mu_y, sigma_sq):
        batch_size = mu_x.size(0)

        a = torch.arange(self.img_height, dtype=mu_x.dtype, device=mu_x.device).view(1, 1, self.img_height)
        b = torch.arange(self.img_width, dtype=mu_x.dtype, device=mu_x.device).view(1, 1, self.img_width)

        FX_unnorm = torch.exp(-torch.pow(a - mu_x.unsqueeze(2), 2) / (2 * sigma_sq.unsqueeze(2)))
        ZX = FX_unnorm.sum(dim=2, keepdim=True)
        FX = FX_unnorm / (ZX + 1e-8)

        FY_unnorm = torch.exp(-torch.pow(b - mu_y.unsqueeze(2), 2) / (2 * sigma_sq.unsqueeze(2)))
        ZY = FY_unnorm.sum(dim=2, keepdim=True)
        FY = FY_unnorm / (ZY + 1e-8)

        return FX, FY

    def _apply_filters(self, x_or_canvas, FX, FY):
        batch_size = x_or_canvas.size(0)
        channels = x_or_canvas.size(1)

        x_reshaped = x_or_canvas.view(batch_size * channels, self.img_height, self.img_width)

        FX_repeated = FX.unsqueeze(1).repeat(1, channels, 1, 1).view(batch_size * channels, self.N, self.img_height)
        FY_repeated = FY.unsqueeze(1).repeat(1, channels, 1, 1).view(batch_size * channels, self.N, self.img_width)

        intermediate_product = torch.bmm(FX_repeated, x_reshaped)

        patch = torch.bmm(intermediate_product, FY_repeated.transpose(1, 2))

        return patch.view(batch_size, channels, self.N, self.N)


    def _predict_attention_params(self, h_dec, is_write_op=False):
        if is_write_op:
            params = self.fc_attn_params_write(h_dec)
        else:
            params = self.fc_attn_params_read(h_dec)

        g_tilde_X, g_tilde_Y, log_sigma_sq, log_delta_tilde, log_gamma_tilde = params.split(1, dim=1)

        g_x = (self.img_height + 1) / 2.0 * (g_tilde_X + 1) - 0.5
        g_y = (self.img_width + 1) / 2.0 * (g_tilde_Y + 1) - 0.5
        delta_norm_factor = (max(self.img_height, self.img_width) - 1.0) / (self.N - 1.0)
        delta = delta_norm_factor * torch.exp(log_delta_tilde)

        sigma_sq = torch.exp(log_sigma_sq)
        gamma = torch.exp(log_gamma_tilde)

        return g_x, g_y, delta, sigma_sq, gamma


    def read_attention(self, x, x_hat, h_dec_prev):
        g_x_read, g_y_read, delta_read, sigma_sq_read, gamma_read = \
            self._predict_attention_params(h_dec_prev, is_write_op=False)

        mu_x_read, mu_y_read = self._get_filter_means(g_x_read, g_y_read, delta_read)
        FX_read, FY_read = self._get_filterbanks(mu_x_read, mu_y_read, sigma_sq_read)

        patch_x = self._apply_filters(x, FX_read, FY_read)
        patch_x_hat = self._apply_filters(x_hat, FX_read, FY_read)

        patch_x_flat = patch_x.view(x.size(0), -1)
        patch_x_hat_flat = patch_x_hat.view(x.size(0), -1)

        concatenated_patches = torch.cat([patch_x_flat, patch_x_hat_flat], dim=1)

        return gamma_read * concatenated_patches

    def write_attention(self, h_dec_curr):
        batch_size = h_dec_curr.size(0)

        wt = self.fc_write_patch(h_dec_curr)
        wt = wt.view(batch_size, self.img_channels, self.N, self.N)

        g_x_write, g_y_write, delta_write, sigma_sq_write, gamma_write = \
            self._predict_attention_params(h_dec_curr, is_write_op=True)

        mu_x_write, mu_y_write = self._get_filter_means(g_x_write, g_y_write, delta_write)
        FX_write, FY_write = self._get_filterbanks(mu_x_write, mu_y_write, sigma_sq_write)

        wt_reshaped = wt.view(batch_size * self.img_channels, self.N, self.N)

        FX_write_repeated = FX_write.unsqueeze(1).repeat(1, self.img_channels, 1, 1).view(batch_size * self.img_channels, self.N, self.img_height)
        FY_write_repeated = FY_write.unsqueeze(1).repeat(1, self.img_channels, 1, 1).view(batch_size * self.img_channels, self.N, self.img_width)
        
        intermediate_write = torch.bmm(FY_write_repeated.transpose(1, 2), wt_reshaped)
        canvas_contribution = torch.bmm(intermediate_write, FX_write_repeated)
        canvas_contribution = canvas_contribution.view(batch_size, self.img_channels, self.img_width, self.img_height).transpose(2,3)

        return (1.0 / (gamma_write.unsqueeze(-1).unsqueeze(-1) + 1e-8)) * canvas_contribution


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_vec, prev_h_enc, prev_c_enc):
        h_enc_curr, c_enc_curr = self.lstm(input_vec, (prev_h_enc, prev_c_enc))
        mu_t = self.fc_mu(h_enc_curr)
        sigma_t = torch.exp(self.fc_sigma(h_enc_curr))
        return h_enc_curr, c_enc_curr, mu_t, sigma_t

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, img_channels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.img_channels = img_channels
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

    def forward(self, latent_sample, prev_h_dec, prev_c_dec):
        h_dec_curr, c_dec_curr = self.lstm(latent_sample, (prev_h_dec, prev_c_dec))
        return h_dec_curr, c_dec_curr
    
class Draw(nn.Module):
    def __init__(self, img_width, img_height, img_channels,
                 enc_hidden_dim, dec_hidden_dim, latent_dim, T, N):
        super().__init__()
        self.img_width    = img_width
        self.img_height   = img_height
        self.img_channels = img_channels
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.latent_dim   = latent_dim
        self.T            = T
        self.N            = N
        self.read_patch_dim = N * N * img_channels * 2

        self.attention = Attention(img_width, img_height, img_channels, N, dec_hidden_dim)
        self.encoder   = Encoder(self.read_patch_dim + dec_hidden_dim, enc_hidden_dim, latent_dim)
        self.decoder   = Decoder(latent_dim, dec_hidden_dim, img_channels)

        self.c_0      = nn.Parameter(torch.zeros(img_channels, img_height, img_width))
        self.h_enc_0  = nn.Parameter(torch.zeros(enc_hidden_dim))
        self.c_enc_0  = nn.Parameter(torch.zeros(enc_hidden_dim))
        self.h_dec_0  = nn.Parameter(torch.zeros(dec_hidden_dim))
        self.c_dec_0  = nn.Parameter(torch.zeros(dec_hidden_dim))

    def forward(self, x):
        B = x.size(0)

        c_t        = self.c_0.unsqueeze(0).expand(B, -1, -1, -1).clone()
        h_enc_prev = self.h_enc_0.unsqueeze(0).expand(B, -1).clone()
        c_enc_prev = self.c_enc_0.unsqueeze(0).expand(B, -1).clone()
        h_dec_prev = self.h_dec_0.unsqueeze(0).expand(B, -1).clone()
        c_dec_prev = self.c_dec_0.unsqueeze(0).expand(B, -1).clone()

        mu_list      = []
        sigma_list   = []
        raw_canvases = []

        for t in range(self.T):
            x_hat_t = x - torch.sigmoid(c_t)

            r_t = self.attention.read_attention(x, x_hat_t, h_dec_prev)

            enc_in = torch.cat([r_t, h_dec_prev], dim=1)
            h_enc_curr, c_enc_curr, mu_t, sigma_t = self.encoder(
                enc_in, h_enc_prev, c_enc_prev
            )

            eps = torch.randn_like(mu_t)
            z_t = mu_t + sigma_t * eps
            h_dec_curr, c_dec_curr = self.decoder(z_t, h_dec_prev, c_dec_prev)

            write_patch = self.attention.write_attention(h_dec_curr)
            c_t = c_t + write_patch

            mu_list.append(mu_t)
            sigma_list.append(sigma_t)
            raw_canvases.append(c_t.clone())
            h_enc_prev = h_enc_curr
            c_enc_prev = c_enc_curr
            h_dec_prev = h_dec_curr
            c_dec_prev = c_dec_curr

        return raw_canvases, mu_list, sigma_list