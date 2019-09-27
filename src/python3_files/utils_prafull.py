def downsample_box_half(x):
    return 0.25 * (
            x[::2, ::2, :] +
            x[1::2, ::2, :] +
            x[1::2, 1::2, :] +
            x[::2, 1::2, :])

def downsample_box_half_5(x):
    return 0.25 * (
            x[:,:,:,::2, ::2] +
            x[:,:,:,1::2, ::2] +
            x[:,:,:,1::2, 1::2] +
            x[:,:,:,::2, 1::2])

def downsample_box_half_4(x):
    return 0.25 * (
            x[:,:,::2, ::2] +
            x[:,:,1::2, ::2] +
            x[:,:,1::2, 1::2] +
            x[:,:,::2, 1::2])


def downsample_box_half_tv(x):
    return 0.25 * (
            x[:,:,::2, ::2,:] +
            x[:,:,1::2, ::2,:] +
            x[:,:,1::2, 1::2,:] +
            x[:,:,::2, 1::2,:])

def downsample_box_half_mono(x):
    return 0.25 * (
            x[::2, ::2] +
            x[1::2, ::2] +
            x[1::2, 1::2] +
            x[::2, 1::2])

def fit_basis_test():
    Z = load_basis()
    Z = torch.from_numpy(Z).float().to(device)

    Zmean = torch.mean(Z, dim=(0, 1))
    Zm = Zmean
    Zm = Zm - Zm.mean()
    Zm = Zm / Zm.std()
    Zmeans = []
    for i in range(4):
        Zmeans.append(Zm.unsqueeze(0).unsqueeze(0))
        Zm = downsample_box_half_mono(Zm)

    tnet = TNet()
    tnet = tnet.to(device)

    #x = tnet(x0)
    #print(x.shape)

    #optimizer = optim.Adam(list(tnet.parameters()) + [x0], lr=0.0001, weight_decay=0.00001, betas=[0.99, 0.999], amsgrad=True) #, momentum=0.9) weight_decay=0.001,
    #optimizer = optim.Adam(list(tnet.parameters()) + [x0], lr=0.001) #, momentum=0.9) weight_decay=0.001,
    optimizer = optim.Adam(list(tnet.parameters()), lr=0.001) #, amsgrad=True)  # , momentum=0.9) weight_decay=0.001,
    print((list(tnet.parameters())))
    #print('parameters', tnet.parameters())
    #optimizer = optim.RMSprop(list(tnet.parameters()) + [x0], lr=0.00031, weight_decay=0.0001)

    #optimizer = optim.Adam(list(tnet.parameters()), lr=0.00021) #, momentum=0.9) weight_decay=0.001,

    criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    #Z = (Z+0.3).log()

    for iter in range(10000000):
        print(iter)
        optimizer.zero_grad()
        T = tnet()

        #T = (T+0.3).log()

        loss = 1*criterion(torch.squeeze(T[:,:,0:-1,:,:,:]), Z)

        loss += criterion(torch.squeeze(T[:,:,1:-1,:,:,:] - T[:,:,0:-2,:,:,:]), Z[1:,:,:,:]-Z[0:-1,:,:,:])
        loss += criterion(torch.squeeze(T[:,:,0:-1,1:,:,:] - T[:,:,0:-1,0:-1,:,:]), Z[:,1:,:,:]-Z[:,0:-1,:,:])
        loss += criterion(torch.squeeze(T[:,:,0:-1,:,1:,:] - T[:,:,0:-1,:,0:-1,:]), Z[:,:,1:,:]-Z[:,:,0:-1,:])
        loss += criterion(torch.squeeze(T[:,:,0:-1,:,:,1:] - T[:,:,0:-1,:,:,0:-1]), Z[:,:,:,1:]-Z[:,:,:,0:-1])

        loss.backward()
        optimizer.step()

        # for conv in pnet.convs:
        #     if np.random.uniform(0, 1.0) < 0.0001:
        #         nn.init.xavier_normal(conv.weight)
        #         conv.bias.data.fill_(0.0001)
        #         print('reset P')
        # for conv in tnet.convs:
        #     if np.random.uniform(0, 1.0) < 0.0001:
        #         nn.init.xavier_normal(conv.weight)
        #         conv.bias.data.fill_(0.001)
        #         print('reset T')


        if iter % 10 == 0:
            print((loss.item()))

        if iter % 30 == 0:
            plt.clf()
            plt.subplot(4,2,1)
            ri = np.random.randint(0,15)
            rj = np.random.randint(0,15)
            #plt.imshow(T.view(Ts[2], Ts[3]).detach().cpu().numpy())
            plt.imshow(np.squeeze(T[0,0,ri,rj,:,:].detach().cpu().numpy()))
            plt.subplot(4,2,2)
            #plt.imshow(T.view(Ts[2], Ts[3]).detach().cpu().numpy())
            plt.imshow(np.squeeze(Z[ri,rj,:,:].detach().cpu().numpy()))
            plt.subplot(4,2,3)
            rx = np.random.randint(3,42)
            ry = np.random.randint(3,42)
            plt.imshow(np.squeeze(T[0,0,:,:,rx,ry].detach().cpu().numpy()))
            plt.subplot(4,2,4)
            plt.imshow(np.squeeze(Z[:,:,rx,ry].detach().cpu().numpy()))
            plt.subplot(4,2,5)
            rt = np.random.randint(1,14)
            plt.imshow(np.squeeze(T[0,0,rt,:,ry,:].detach().cpu().numpy()))
            plt.subplot(4,2,6)
            plt.imshow(np.squeeze(Z[rt,:,ry,:].detach().cpu().numpy()))
            plt.subplot(4,2,7)
            plt.imshow(np.squeeze((T[0,0,rt,:,ry,:]-T[0,0,rt+1,:,ry,:]).detach().cpu().numpy()))
            plt.subplot(4,2,8)
            plt.imshow(np.squeeze((Z[rt,:,ry,:]-Z[rt+1,:,ry,:]).detach().cpu().numpy()))

            fig.canvas.draw()
            fig.canvas.flush_events()

def fit_video_test():
    V = load_video()
    V = torch.from_numpy(V).float().to(device)

    #V = V + torch.randn(V.shape, device=device) * 0.5
    nframes = V.shape[0]

    vnet = VNet(nframes)
    #vnet = VNet_Alt(nframes)
    vnet = vnet.to(device)

    x = vnet()
    print((x.shape))

    optimizer = optim.Adam(list(vnet.parameters()), lr=0.0001) #, amsgrad=True)  # , momentum=0.9) weight_decay=0.001,

    criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    #V = (V+0.1).log()

    for iter in range(10000000):
        print(iter)
        optimizer.zero_grad()
        v = vnet()
        v = torch.squeeze(v)    # XXX color
        #T = (T+0.3).log()

        loss = 0.3*criterion(torch.squeeze(v), V)
        loss += 1*criterion(v[1:,:,:] - v[:-1,:,:], V[1:,:,:] - V[:-1,:,:])



        #loss += criterion(torch.squeeze(T[:,:,1:-1,:,:,:] - T[:,:,0:-2,:,:,:]), Z[1:,:,:,:]-Z[0:-1,:,:,:])
        #loss += criterion(torch.squeeze(T[:,:,0:-1,1:,:,:] - T[:,:,0:-1,0:-1,:,:]), Z[:,1:,:,:]-Z[:,0:-1,:,:])
        #loss += criterion(torch.squeeze(T[:,:,0:-1,:,1:,:] - T[:,:,0:-1,:,0:-1,:]), Z[:,:,1:,:]-Z[:,:,0:-1,:])
        #loss += criterion(torch.squeeze(T[:,:,0:-1,:,:,1:] - T[:,:,0:-1,:,:,0:-1]), Z[:,:,:,1:]-Z[:,:,:,0:-1])

        loss.backward()
        optimizer.step()

        # for conv in pnet.convs:
        #     if np.random.uniform(0, 1.0) < 0.0001:
        #         nn.init.xavier_normal(conv.weight)
        #         conv.bias.data.fill_(0.0001)
        #         print('reset P')
        # for conv in tnet.convs:
        #     if np.random.uniform(0, 1.0) < 0.0001:
        #         nn.init.xavier_normal(conv.weight)
        #         conv.bias.data.fill_(0.001)
        #         print('reset T')


        if iter % 10 == 0:
            print((loss.item()))

        if iter % 1000 == 0:
            print((loss.item()))

        if iter % 30 == 0:
            plt.clf()
            rf = np.random.randint(0,V.shape[0]-1)
            plt.subplot(3,2,1)
            plt.imshow(np.squeeze(V[rf,:,:].detach().cpu().numpy()))
            plt.subplot(3,2,2)
            plt.imshow(np.squeeze(v[rf,:,:].detach().cpu().numpy()))
            plt.subplot(3, 2, 3)
            plt.imshow(np.transpose(np.squeeze(V[:,10,:].detach().cpu().numpy())))
            plt.subplot(3, 2, 4)
            plt.imshow(np.transpose(np.squeeze(v[:,10,:].detach().cpu().numpy())))
            plt.subplot(3, 2, 5)
            #plt.imshow(np.transpose(np.squeeze(vnet.xi.detach().cpu().numpy())))
            plt.subplot(3, 2, 6)
            #plt.plot((np.squeeze(vnet.xi.detach().cpu().numpy())))
            fig.canvas.draw()
            fig.canvas.flush_events()