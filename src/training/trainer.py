#
#
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         logits = model.forward(batch)
#         loss = criterion.forward(logits, targets)
#         delta_Z = criterion.backward()
#         model.backward(delta_Z)
#         optimizer.step(model.params(), model.grads())
