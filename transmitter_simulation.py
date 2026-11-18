"""
Simulated network channel between client and server (no HTTP).
Only tensors cross the boundary: the sender's graph is cut with .detach(); on the way back
the receiver's gradient is handed to the sender, which resumes back-propagation locally.

Traffic accounting (v3): every channel counts its forward payload (activations, sender -> receiver)
and its backward payload (gradients, receiver -> sender) separately, so that the traffic per
direction can be reconstructed exactly:
    client->server = up.fwd_bytes + down.bwd_bytes
    server->client = up.bwd_bytes + down.fwd_bytes
(v2 added the client->server gradient to the 'down' counter and never reported the
server->client gradient.)
"""
import torch


class SplitChannel:
    def __init__(self, name=""):
        self.name = name
        self._recv = None
        self.fwd_bytes = 0
        self.bwd_bytes = 0

    def send(self, tensor, direction=None):
        """Cut the graph and return the tensor as the receiver sees it."""
        self._recv = tensor.detach().clone().requires_grad_(tensor.requires_grad)
        self.fwd_bytes += tensor.numel() * tensor.element_size()
        return self._recv

    def send_grad_back(self):
        """Gradient that the receiver computed w.r.t. what it received."""
        g = self._recv.grad
        self.bwd_bytes += g.numel() * g.element_size()
        return g

    def reset(self):
        self.fwd_bytes = self.bwd_bytes = 0
