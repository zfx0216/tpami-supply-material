import torch
import torch.nn as nn

from attack import Attack


class MIFGSM(Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

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

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()

            delta = adv_images - images
            delta[0][0] = torch.clamp(adv_images[0][0] - images[0][0], min=-0.13, max=0.13)
            delta[0][1] = torch.clamp(adv_images[0][1] - images[0][1], min=-0.13, max=0.13)
            delta[0][2] = torch.clamp(adv_images[0][2] - images[0][2], min=-0.13, max=0.13)

            adv_images[0][0] = torch.clamp(images[0][0] + delta[0][0], min=-2.11, max=2.24).detach()
            adv_images[0][1] = torch.clamp(images[0][1] + delta[0][1], min=-2.03, max=2.42).detach()
            adv_images[0][2] = torch.clamp(images[0][2] + delta[0][2], min=-1.80, max=2.64).detach()

        return adv_images