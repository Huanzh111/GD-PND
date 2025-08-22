import torch.nn as nn

import torch


class GAM_Attention(nn.Module):

    def __init__(self, in_channels, out_channels, rate=4):

        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(

            nn.Linear(in_channels, int(in_channels / rate)),

            nn.ReLU(inplace=True),

            nn.Linear(int(in_channels / rate), in_channels)

        )


        self.spatial_attention = nn.Sequential(

            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),

            nn.BatchNorm2d(int(in_channels / rate)),

            nn.ReLU(inplace=True),

            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),

            nn.BatchNorm2d(out_channels)

        )
        self.mlp = nn.Sequential(

            nn.Linear(in_channels, int(in_channels / rate)),

            nn.ReLU(inplace=True),

            nn.Linear(int(in_channels / rate), out_channels)

        )

        # self.mlp_2 = nn.Sequential(
        #
        #     nn.Linear(in_channels, int(in_channels / rate)),
        #
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(int(in_channels / rate), out_channels)
        #
        # )

    def forward(self, x):
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        b, c, h, w = x.shape

        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)

        # x_permute_1 = self.mlp_2(x_permute)
        # x_permute = x_permute + 0.2 * x_permute_1

        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        # x_per = x.permute(0, 2, 3, 1).view(b, -1, c)
        # x_per = self.mlp_2(x_per).view(b, h, w, c).permute(0, 3, 1, 2)
        # x = x + 0.2 * x_per

        x_spatial_att = self.spatial_attention(x).sigmoid()

        out = x * x_spatial_att

        out_per = out.permute(0, 2, 3, 1).view(b, -1, c)
        out_mlp = self.mlp(out_per).view(b, h, w, c).permute(0, 3, 1, 2)
        out = out + 0.2 * out_mlp



        return out
#
# if __name__ == '__main__':
#
#     x = torch.randn(1, 32, 128, 128)
#
#     b, c, h, w = x.shape
#
#     net = GAM_Attention(in_channels=c, out_channels=c)
#     y = net.forward(x)
#     print(y.shape)