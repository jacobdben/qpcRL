

# bounds=((19,41),(25,45))#((0,60),(0,70))
# fig,ax=plt.subplots() 
# QPC.U0=disorder
# ax.set_title("Optimized") 
# QPC.set_all_pixels(x+common_voltages[14])
# t1,p1=QPC.plot_potential_section(bounds=bounds,ax=ax)
# plt.colorbar(p1)
# plt.savefig(figurepath+"2")

# fig,ax=plt.subplots()
# ax.set_title("Non Optimized")
# QPC.U0=disorder
# QPC.set_all_pixels(common_voltages[14])
# t2,p2=QPC.plot_potential_section(bounds=bounds,ax=ax)
# plt.colorbar(p2)
# # np.where(t1==t2)
# plt.savefig(figurepath+"3")

# fig,ax=plt.subplots()
# ax.set_title("Non Optimized,no disorder")
# QPC.U0=0
# QPC.set_all_pixels(common_voltages[14])
# t3,p3=QPC.plot_potential_section(bounds=bounds,ax=ax)
# plt.colorbar(p3)
# plt.savefig(figurepath+"4")
# # np.where(t1==t2)

# fig,ax=plt.subplots()
# ax.set_title('Non Optimized - NonOpt No Disorder')
# p4=ax.imshow(t2.T-t3.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
# plt.colorbar(p4)
# plt.savefig(figurepath+"5")

# fig,ax=plt.subplots()
# ax.set_title('Optimized - NonOpt No Disorder')
# p5=ax.imshow(t1.T-t3.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
# plt.colorbar(p5)
# plt.savefig(figurepath+"6")


# fig,ax=plt.subplots()
# ax.set_title('Optimized - NonOpt')
# p6=ax.imshow(t1.T-t2.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
# plt.colorbar(p6)
# plt.savefig(figurepath+"7")