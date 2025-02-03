import torch
import torch.nn as nn
import torch.optim as optim

from attack import Attack


class CW(Attack):
    def __init__(self, model, c=1, kappa=0, steps=50, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        w = images.clone().detach().to(self.device)
        w[0][0] = self.inverse_tanh_space_R(images[0][0]).detach()
        w[0][1] = self.inverse_tanh_space_G(images[0][1]).detach()
        w[0][2] = self.inverse_tanh_space_B(images[0][2]).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)
            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                condition = (pre == target_labels).float()
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images


    def tanh_space(self, x):
        x_clone = x.clone()

        x_clone[0][0] = 0.5 * (torch.tanh(x[0][0]) + 1) * (2.24 + 2.11) - 2.11

        x_clone[0][1] = 0.5 * (torch.tanh(x[0][1]) + 1) * (2.42 + 2.03) - 2.03

        x_clone[0][2] = 0.5 * (torch.tanh(x[0][2]) + 1) * (2.64 + 1.8) - 1.8

        return x_clone

    # def inverse_tanh_space(self, x):
    #     # torch.atanh is only for torch >= 1.7.0
    #     # atanh is defined in the range -1 to 1
    #     return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def inverse_tanh_space_R(self, x):
        converted_value = (x + 2.11) / (2.24 + 2.11) * 2 - 1
        return self.atanh(torch.clamp(converted_value, min=-1, max=1))

    def inverse_tanh_space_G(self, x):
        converted_value = (x + 2.03) / (2.42 + 2.03) * 2 - 1
        return self.atanh(torch.clamp(converted_value, min=-1, max=1))

    def inverse_tanh_space_B(self, x):
        converted_value = (x + 1.80) / (2.64 + 1.80) * 2 - 1
        return self.atanh(torch.clamp(converted_value, min=-1, max=1))


    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)
