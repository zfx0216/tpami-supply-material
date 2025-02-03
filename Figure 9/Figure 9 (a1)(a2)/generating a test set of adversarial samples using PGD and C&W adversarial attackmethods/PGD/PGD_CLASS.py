import torch
import torch.nn as nn

from attack import Attack


class PGD(Attack):
    def __init__(self, model, eps=8 / 255, alpha=1 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            delta = adv_images - images
            delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.13, max=0.13)
            delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.13, max=0.13)
            delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.13, max=0.13)

            adv_images[0][0] = torch.clamp(images[0][0] + delta[0][0], min=-2.11, max=2.24).detach()
            adv_images[0][1] = torch.clamp(images[0][1] + delta[0][1], min=-2.03, max=2.42).detach()
            adv_images[0][2] = torch.clamp(images[0][2] + delta[0][2], min=-1.80, max=2.64).detach()

        return adv_images
