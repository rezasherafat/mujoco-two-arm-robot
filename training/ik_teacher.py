"""Optional analytic teaching signal for PPO; never used by browser inference."""
import torch


def fk(q: torch.Tensor) -> torch.Tensor:
    a, b = q.unbind(-1)
    return torch.stack((.6*a.sin()+.5*(a+b).sin(), .6*a.cos()+.5*(a+b).cos()), -1)


@torch.no_grad()
def teacher_actions(observations: torch.Tensor) -> torch.Tensor:
    q = torch.stack((torch.atan2(observations[:, 0], observations[:, 1]),
                     torch.atan2(observations[:, 2], observations[:, 3])), -1)
    target = observations[:, 6:8] + fk(q)
    cosine = ((target.square().sum(-1) - .6**2 - .5**2) / (.6)).clamp(-1, 1)
    angle = cosine.acos()
    q2 = torch.stack((angle, -angle), -1)
    q1 = torch.atan2(target[:, 0], target[:, 1])[:, None] - torch.atan2(
        .5 * q2.sin(), .6 + .5 * q2.cos())
    solutions = torch.stack((q1, q2), -1)
    distance = (solutions - q[:, None]).square().sum(-1)
    distance = distance.masked_fill((solutions.abs() > 2.8 + 1e-5).any(-1), float("inf"))
    selected = solutions[torch.arange(len(q), device=q.device), distance.argmin(-1)]
    return ((selected - q) / .4).clamp(-1, 1)


@torch.no_grad()
def teaching_batch(count: int, device: torch.device):
    q = torch.empty((count, 2), device=device).uniform_(-2.8, 2.8)
    goal = torch.empty_like(q).uniform_(-2.8, 2.8)
    # Deliberately cover precise settling as well as long reaches.
    goal[:count//3] = (q[:count//3] + .05 * torch.randn_like(q[:count//3])).clamp(-2.8, 2.8)
    goal[count//3:2*count//3] = (q[count//3:2*count//3] + .4 * torch.randn_like(q[count//3:2*count//3])).clamp(-2.8, 2.8)
    obs = torch.cat((q[:, 0:1].sin(), q[:, 0:1].cos(), q[:, 1:2].sin(), q[:, 1:2].cos(),
                     .2 * torch.randn_like(q), fk(goal) - fk(q),
                     .2 * torch.randn_like(q)), -1)
    return obs, teacher_actions(obs)
