## Question: L1 and L2 Regularization — Formula, Geometry, and Intuition

In linear regression, regularization helps prevent overfitting by adding penalty terms to the loss function. Let $\mathcal{L}_{train}(W)$ be the original data loss and $\lambda > 0$ be the regularization strength.

1. Write the complete objective function for linear regression with L1 regularization (Lasso).
2. Write the complete objective function for linear regression with L2 regularization (Ridge).
3. Which regularization method (L1 or L2) can produce exactly zero weights? Which one tends to shrink weights smoothly?
4. In 2D space with weights $w_1$ and $w_2$, draw two figures:
   - First figure: Show data loss contours (ellipses) and the L1 constraint region. Mark where a contour first touches the constraint region.
   - Second figure: Show data loss contours (ellipses) and the L2 constraint region. Mark where a contour first touches the constraint region.
5. Based on your drawings, explain why one method leads to sparse solutions (feature selection) while the other does not.
6. Why is $\lambda$ considered a hyperparameter that needs tuning?