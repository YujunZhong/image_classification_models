import torch
from utils import cross_entropy_loss
import unittest
from mlp import Linear, MLP
from resnet18 import BasicBlock, ResNet18
from mlpmixer import PatchEmbed, MixerBlock, MLPMixer

class TestLinear(unittest.TestCase):
    def test_linear_attributes(self):
        in_feat = 30
        out_feat = 20
        my_linear = Linear(in_features=in_feat, out_features=out_feat)
        assert hasattr(my_linear, 'weight')
        assert hasattr(my_linear, 'bias')
        
        assert len(my_linear.weight.shape) == 2
        assert my_linear.weight.shape[0] == out_feat
        assert my_linear.weight.shape[1] == in_feat
        
        assert len(my_linear.bias.shape) == 1
        assert my_linear.bias.shape[0] == out_feat
    
    def test_linear_forward(self):
        in_feat = 30
        out_feat = 20
        my_linear = Linear(in_features=in_feat, out_features=out_feat)
        
        gt_linear = torch.nn.Linear(in_features=in_feat, out_features=out_feat)
        my_linear.weight.data[:] = gt_linear.weight.data
        my_linear.bias.data[:] = gt_linear.bias.data
        
        batch = 10
        inputs = torch.randn(batch, in_feat)
        my = my_linear(inputs)
        assert len(my.shape) == 2
        assert my.shape[0] == batch
        assert my.shape[1] == out_feat
        
        gt = gt_linear(inputs)
        assert torch.allclose(my, gt)
        
class TestMLP(unittest.TestCase):
    input_size = 50
    hidden_sizes = [100, 200]
    output_size = 20
    batch = 10
    
    def test_mlp(self):
        model = MLP(self.input_size, self.hidden_sizes, self.output_size)
        assert len(model.hidden_layers) == len(self.hidden_sizes)
        
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for layer_id, layer in enumerate(model.hidden_layers + [model.output_layer]):
            assert isinstance(layer, Linear)
            in_feat = sizes[layer_id]
            out_feat = sizes[layer_id + 1]
            assert layer.weight.shape[0] == out_feat
            assert layer.weight.shape[1] == in_feat
    
    def test_activation(self):
        model = MLP(self.input_size, self.hidden_sizes, self.output_size)
        inputs = torch.randn(self.batch, self.input_size)
        
        names = ['relu', 'tanh', 'sigmoid']
        gtfuncs = [
            torch.relu, 
            torch.tanh, 
            torch.sigmoid]
        
        for activation_name, gtfunc in zip(names, gtfuncs):
            gt = gtfunc(inputs)
            my = model.activation_fn(activation_name, inputs)
            assert torch.allclose(my, gt)
    
    def test_forward(self):     
        model = MLP(self.input_size, self.hidden_sizes, self.output_size)
        inputs = torch.randn(self.batch, self.input_size)
        outputs = model(inputs)
        assert len(outputs.shape) == 2
        assert outputs.shape[0] == self.batch
        assert outputs.shape[1] == self.output_size

class TestResNet(unittest.TestCase):
    def test_basic_block(self):
        block = BasicBlock(64, 64, 1)
        inputs = torch.randn(32, 64, 8, 8)
        outputs = block(inputs)
        assert len(outputs.shape) == 4
        assert outputs.shape[0] == 32
        assert outputs.shape[1] == 64
        assert outputs.shape[2] == 8
        assert outputs.shape[3] == 8

    def test_basic_block2(self):
        block = BasicBlock(64, 128, 1)
        inputs = torch.randn(32, 64, 8, 8)
        outputs = block(inputs)
        assert len(outputs.shape) == 4
        assert outputs.shape[0] == 32
        assert outputs.shape[1] == 128
        assert outputs.shape[2] == 8
        assert outputs.shape[3] == 8

    def test_resnet(self):
        model = ResNet18(10)
        inputs = torch.randn(50, 3, 32, 32)
        logits = model(inputs)
        assert len(logits.shape) == 2
        assert logits.shape[0] == 50
        assert logits.shape[1] == 10

class TestMLPMixer(unittest.TestCase):
    embed_dim = 512
    img_size = 32
    patch_size = 4
    batch_size = 10
    
    def test_patch_emb(self):
        mod = PatchEmbed(self.img_size, self.patch_size, 3, self.embed_dim)
        test_img = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        out = mod(test_img)
        assert len(out.shape) == 3
        assert out.shape[0] == self.batch_size
        assert out.shape[1] == mod.num_patches
        assert out.shape[2] == self.embed_dim
    
    def test_mixer_block(self):
        num_patches = self.img_size // self.patch_size
        seq_len = num_patches**2
        mod = MixerBlock(dim=self.embed_dim, seq_len=seq_len)
        inputs = torch.randn(self.batch_size, seq_len, self.embed_dim)
        output = mod(inputs)
        assert len(output.shape) == 3
        assert output.shape[0] == self.batch_size
        assert output.shape[1] == seq_len
        assert output.shape[2] == self.embed_dim

    def test_mlpmixer(self):
        model = MLPMixer(num_classes=10, 
                         img_size=self.img_size, 
                         patch_size=self.patch_size,
                         embed_dim=self.embed_dim,
                         num_blocks=4)
        inputs = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        outputs = model(inputs)
        assert len(outputs.shape) == 2
        assert outputs.shape[0] == self.batch_size
        assert outputs.shape[1] == 10
        
class TestUtils(unittest.TestCase):
    def test_ce_loss(self):
        gt_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        ce_loss = cross_entropy_loss
    
        batch_size = 30
        label_size = 10
    
        for _ in range(5):
            logits = torch.randn(batch_size, label_size)
            labels = torch.randint(label_size, size=[batch_size])
            gt = gt_loss(logits, labels)
            ce = ce_loss(logits, labels)
            assert torch.allclose(gt, ce), "test_ce_loss failed"


if __name__ == '__main__':
    unittest.main(verbosity=2)