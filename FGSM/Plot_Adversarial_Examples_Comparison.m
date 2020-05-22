X1 = [2767,2995,3730,4610,5712];
X2 = [673,549,254,251,237];
Y1 = [1:5];

figure();
subplot(2,2,1);
plot(Y1,X1,'-bo','LineWidth',2);
title('Attack Generator - FGSM Attack Strength');
xlabel('Iteration');
ylabel('Fooling Adversarial Inputs (Y/9996)');

subplot(2,2,2);
plot(Y1,X2,'-ko','LineWidth',2);
title('Attack Generator - FGSM Defense Strength');
xlabel('Iteration');
ylabel('Fooling Adversarial Inputs (Y/9996)');

X3 = [2190,2838,3176,3796,3959];
X4 = [796,330,253,237,229];
Y2 = [1:5];

subplot(2,2,3);
plot(Y2,X3,'-bo','LineWidth',2);
title('Attack Generator - DeepFool Attack Strength');
xlabel('Iteration');
ylabel('Fooling Adversarial Inputs (Y/6773)');

subplot(2,2,4);
plot(Y2,X4,'-ko','LineWidth',2);
title('Attack Generator - DeepFool Defense Strengths');
xlabel('Iteration');
ylabel('Fooling Adversarial Inputs (Y/6773)');