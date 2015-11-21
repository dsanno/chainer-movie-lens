import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import Variable
from model import Model

class MovieLensModel(Model):
    def __init__(self, user_size, item_size):
        Model.__init__(self,
            in_user = F.EmbedID(user_size, 100),
            in_item = F.EmbedID(item_size, 100),
            layer1 = F.Linear(100, 100),
            layer2 = F.Linear(100, 100),
            layer3 = F.Linear(100, 1)
        )
        self.user_size = user_size
        self.item_size = item_size

    def forward(self, (x_user, x_item), train=True):
        xp = cuda.get_array_module(x_user.data)
        h0 = F.dropout(F.relu(self.in_user(x_user) + self.in_item(x_item)), train=train)
        h1 = F.dropout(F.relu(self.layer1(h0)), train=train)
        h2 = F.dropout(F.relu(self.layer2(h1)), train=train)
        h3 = F.sigmoid(self.layer3(h2)) * 5.0 + 0.5
        return h3
