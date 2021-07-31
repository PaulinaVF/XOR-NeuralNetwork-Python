# Red neuronal solucionar XOR
import math
from tkinter.filedialog import asksaveasfile

coef_Aprendizaje = 0.1
trainingSamples = [[[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]]
n_in = 2
n_hid = 2
n_out = 1
# Los pesos se van a guardar en dos listas diferentes, la que afecta la entrada de la capa oculta y la que afecta la entrada de la capa de salida
# Para respetar los índices y no perder de dónde a dónde se considera el peso se trabajarán como "matrices" para representar los índices i y j
w_hidInput = [[0.0 for i in range(n_hid)] for j in range(n_in+1)]
w_outInput = [[0.0 for i in range(n_out)] for j in range(n_hid+1)]
w_hidInput[0][0] = 0.8
w_hidInput[1][0] = 0.5
w_hidInput[2][0] = 0.4
w_hidInput[0][1] = -0.1
w_hidInput[1][1] = 0.9
w_hidInput[2][1] = 1
w_outInput[0][0] = 0.3
w_outInput[1][0] = -1.2
w_outInput[2][0] = 1.1

iteracion = 0
errorFound = True
while(errorFound):
    iteracion += 1
    # Inicia asumiendo que ya no hay error
    errorFound = False
    allOuts = [] # Para guardar las soluciones encontradas
    for i_sample in range(len(trainingSamples[0])):
        f_hid, o_hid = [0.0 for i in range(n_hid)], [0.0 for i in range(n_hid)]
        f_out, o_out= [0.0 for i in range(n_out)], [0.0 for i in range(n_out)]

        # Forward
        for i in range(len(f_hid)):
            f_hid[i] += (-1*w_hidInput[0][i])
            j = 0
            while (j < n_in):
                f_hid[i] += trainingSamples[0][i_sample][j]*w_hidInput[j+1][i]
                j += 1
        for i in range(len(o_hid)):
            o_hid[i] = 1/(1+math.exp(-f_hid[i]))

        for i in range(len(f_out)):
            f_out[i] += (-1 * w_outInput[0][i])
            j = 0
            while (j < n_hid):
                f_out[i] += o_hid[j] * w_outInput[j+1][i]
                j += 1
        for i in range(len(o_out)):
            o_out[i] = 1/(1+math.exp(-f_out[i]))

        # Comprobar si hubo error
        addtoall = []
        for i in range(len(o_out)):
            addtoall.append(round(o_out[i],4))
            if(abs(o_out[i] - trainingSamples[1][i_sample][i])>0.2):
                errorFound = True
                esteOutBien = False
            else:
                esteOutBien = True
        allOuts.append(addtoall)

        # Backward
        if not esteOutBien:
            error_out = [0.0 for i in range(n_out)]
            error_hid = [0.0 for i in range(n_hid)]

            for i in range(n_out):
                error_out[i] = o_out[i]*(1-o_out[i])*(trainingSamples[1][i_sample][i]-o_out[i])

            for i in range(n_hid):
                sum_pesoError = 0.0
                for j in range(n_out):
                    sum_pesoError += (w_outInput[i+1][j] * error_out[j])
                error_hid[i] = o_hid[i]*(1-o_hid[i])*sum_pesoError

            for j in range(n_hid):
                w_hidInput[0][j] += (coef_Aprendizaje*error_hid[j]*(-1))
                for i in range(n_in):
                    w_hidInput[i+1][j] += (coef_Aprendizaje * error_hid[j] * trainingSamples[0][i_sample][i])

            for j in range(n_out):
                w_outInput[0][j] += (coef_Aprendizaje*error_out[j]*-1)
                for i in range(n_hid):
                    w_outInput[i+1][j] += (coef_Aprendizaje * error_out[j] * o_hid[i])


roundedOuts = [[0] for i in range(len(allOuts))]
for i in range(len(roundedOuts)):
    roundedOuts[i][0] = round(allOuts[i][0])

print('En '+str(iteracion)+' iteraciones se encuentra:')
for i in range(len(trainingSamples[0])):
    print('Entrada: '+str(trainingSamples[0][i])+' | Salida real: '+str(trainingSamples[1][i])+' | Salida encontrada: '+str(allOuts[i])+' -> Se considera:'+str(roundedOuts[i]))


# Para guardar los pesos al terminar el entrenamiento
file = asksaveasfile(mode='w', defaultextension=".txt")
if (file is not None):
    for row in w_hidInput:
        for element in row:
            file.write(str(element) + ' ')
        file.write('\n')
    file.close()