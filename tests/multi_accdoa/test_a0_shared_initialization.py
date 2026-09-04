import importlib.util
import sys

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old_root = '/work/zhanghc/Myllm/SELD/CausalMultiACCDOA_TAU2020'
new_root = '/work/zhanghc/Myllm/SELD/DynamicCausalMultiACCDOA_TAU2020'
old_parameters = load_module('a0_parameters', old_root + '/parameters.py')
old_model_module = load_module('a0_seldnet_model', old_root + '/seldnet_model.py')
new_parameters = load_module('dynamic_parameters', new_root + '/parameters.py')
new_model_module = load_module('dynamic_seldnet_model', new_root + '/seldnet_model.py')

old_params = old_parameters.get_params('32')
new_params = new_parameters.get_params('50')
assert not old_params['causal'] and not new_params['causal']

in_shape = (2, 7, 25, 64)
out_shape = (2, 5, 126)
torch.manual_seed(2026)
old_model = old_model_module.SeldModel(in_shape, out_shape, old_params).eval()
torch.manual_seed(2026)
new_model = new_model_module.SeldModel(in_shape, out_shape, new_params).eval()

old_state = old_model.state_dict()
new_state = new_model.state_dict()
for key, value in old_state.items():
    assert key in new_state, key
    assert torch.equal(value, new_state[key]), key

x = torch.randn(2, 7, 25, 64)
with torch.no_grad():
    old_output = old_model(x)
    new_output = new_model(x)['doa']
assert torch.equal(old_output, new_output)
print('A0_SHARED_INITIALIZATION_PASS', len(old_state))
