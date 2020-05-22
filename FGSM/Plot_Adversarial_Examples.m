X1 = [2767,2995,3730,4610,5712];
X2 = [673,549,254,251,237];
Y = [1:5];

figure();
plot(Y,X1,'-bo','LineWidth',2);
title('Attacking Strength of Pix2Pix-Generated Adversarial Examples');
xlabel('Iteration');
ylabel('Number of Fooling Adversarial Inputs (Y/9996)');

figure();
plot(Y,X2,'-ko','LineWidth',2);
title('Defense Strength of Pix2Pix-Generated Adversarial Examples');
xlabel('Iteration');
ylabel('Number of Fooling Adversarial Inputs (Y/9996)');
