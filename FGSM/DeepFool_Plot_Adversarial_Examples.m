X1 = [2190,2838,3176,3796];
X2 = [796,330,253,237];
Y = [1:4];

figure();
plot(Y,X1,'-bo','LineWidth',2);
title('Attacking Strength of Pix2Pix-Generated Adversarial Examples');
xlabel('Iteration');
ylabel('Number of Fooling Adversarial Inputs (Y/6773)');

figure();
plot(Y,X2,'-ko','LineWidth',2);
title('Defense Strength of Pix2Pix-Generated Adversarial Examples');
xlabel('Iteration');
ylabel('Number of Fooling Adversarial Inputs (Y/6773)');
