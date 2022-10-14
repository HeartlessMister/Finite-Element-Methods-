if element_type == "D2TR8N":
     EL = np.hstack([EL, np.zeros([EL, 4])])

     for i in range(1, NoE + 1):
         data = NL[EL[i, [0, 1, 2, 3]] - 1, :]
         avg_1 = (data[0] + data[1]) / 2
         avg_2 = (data[1] + data[2]) / 2
         avg_3 = (data[2] + data[3]) / 2
         avg_4 = (data[3] + data[0]) / 2

         try:
             index1 = np.where(np.all(NL == avg_1, axis=1))[0][0]
         except:
             index1 = np.size(NL, 0)

         try:
             index2 = np.where(np.all(NL == avg_2, axis=1))[0][0]
         except:
             index2 = np.size(NL, 0)

         try:
             index3 = np.where(np.all(NL == avg_3, axis=1))[0][0]
         except:
             index3 = np.size(NL, 0)

         try:
             index4 = np.where(np.all(NL == avg_4, axis=1))[0][0]
         except:
             index4 = np.size(NL, 0)

         if index1:
             EL[i, 5] = index1
         else:
             EL[i, 5] = index1
             NL = np.vstack([NL, avg_1])

         if index2:
             EL[i, 5] = index2
         else:
             EL[i, 5] = index2
             NL = np.vstack([NL, avg_2])

         if index3:
             EL[i, 5] = index3
         else:
             EL[i, 5] = index3
             NL = np.vstack([NL, avg_3])

         if index4:
             EL[i, 5] = index4
         else:
             EL[i, 5] = index4
             NL = np.vstack([NL, avg_4])




     NPE_new = 3
     NoE_new = 2 * NoE
     EL_new = np.zeros([NoE_new, NPE_new])

     for i in range(1, NoE + 1):
         EL_new[2 * (i - 1), 0] = EL[i - 1, 0]
         EL_new[2 * (i - 1), 1] = EL[i - 1, 1]
         EL_new[2 * (i - 1), 2] = EL[i - 1, 2]

         EL_new[2 * (i - 1) + 1, 0] = EL[i - 1, 0]
         EL_new[2 * (i - 1) + 1, 1] = EL[i - 1, 2]
         EL_new[2 * (i - 1) + 1, 2] = EL[i - 1, 3]

     EL = EL_new

    EL = EL.astype(int)

    return NL, EL