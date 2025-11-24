from infra import get_tensor, get_tensor_model, get_tensor_interface, shape_interface

def construct_buffer(h_grid, v_grid):
  # struct to retain duplicate info so explicit tendency 
  # can be split into multiple routines.
  buffer = {"phi": 
            "phi_i": ,
            "v_i": ,
            "dpi_i": }
  return buffer

def explicit_tend_hydro(state, h_grid, v_grid):
  dpi_i = get_tensor_interface(h_grid, v_grid)
  phi_i = get_tensor_interface(h_grid, v_grid)
  phi = get_tensor_model(h_grid, v_grid)
  dpi = state["dpi"]
  u = state["u"]
  vtheta_dpi = state["vtheta_dpi"]
  vtheta = vtheta_dpi/dpi
  dpi_i[:,:,:,-1] = state["dpi"][:,:,:,-1]
  dpi_i[:,:,:,0] = state["dpi"][:,:,:,0]
  dpi_i[:,:,:,1:-1] = 0.5 * (state["dpi"][:,:,:,1:] + state["dpi"][:,:,:,:-1])

  # dp3d_i(:,:,1) = dp3d(:,:,1)
  # dp3d_i(:,:,nlevp) = dp3d(:,:,nlev)
  # do k=2,nlev
  #    dp3d_i(:,:,k)=(dp3d(:,:,k)+dp3d(:,:,k-1))/2
  # end do
  u_i = get_tensor((*shape_interface(h_grid, v_grid), 2))
  u_i[:,:,:,0, :] = u[:,:,:,0,:] 
  u_i[:,:,:,-1, :] = u[:,:,:,-1,:] 
  u_i[:,:,:,1:-1,:] = (dpi[:,:,:,1:, np.newaxis] * u[:,:,:,1:,:] + 
                       dpi[:,:,:,:-1, np.newaxis] * u[:,:,:,:-1,:])/ (2.0 * dpi_i[:,:,:,1:])
  # ! special averaging for velocity for energy conservation
  # v_i(:,:,1:2,1) = elem(ie)%state%v(:,:,1:2,1,n0)  
  # v_i(:,:,1:2,nlevp) = elem(ie)%state%v(:,:,1:2,nlev,n0)
  # do k=2,nlev
  #    v_i(:,:,1,k) = (dp3d(:,:,k)*elem(ie)%state%v(:,:,1,k,n0) + &
  #         dp3d(:,:,k-1)*elem(ie)%state%v(:,:,1,k-1,n0) ) / (2*dp3d_i(:,:,k))
  #    v_i(:,:,2,k) = (dp3d(:,:,k)*elem(ie)%state%v(:,:,2,k,n0) + &
  #         dp3d(:,:,k-1)*elem(ie)%state%v(:,:,2,k-1,n0) ) / (2*dp3d_i(:,:,k))
  # end do
  phi_i = get_balanced_phi(state, h_grid, v_grid)
  # do k=nlev,1,-1          ! traditional Hydrostatic integral
  #    phi_i(:,:,k)=phi_i(:,:,k+1)+&
  #         Rgas*vtheta_dp(:,:,k)*exner(:,:,k)/pnh(:,:,k)
  # enddo
  phi = 0.5 * (phi_i[:,:,:,1:] + phi_i[:,:,:,:-1])
  #phi(:,:,k) = (phi_i(:,:,k)+phi_i(:,:,k+1))/2  ! for diagnostics
  divdp = sphere_divergence_nlev(dpi[:,:,:,:,np.newaxis] * u, h_grid, a= earth_radius)
  # divdp(:,:,k)=divergence_sphere(vtemp(:,:,:,k),deriv,elem(ie))
  vort = sphere_vorticity_nlev(u, h_grid, a=earth_radius)
  # vort(:,:,k)=vorticity_sphere(elem(ie)%state%v(:,:,:,k,n0),deriv,elem(ie))

  # ! compute gradphi at interfaces and then average to levels
  grad_phi = sphere_gradient_nlev(phi_i, h_grid, a=earth_radius)
  # gradphinh_i(:,:,:,k)   = gradient_sphere(phi_i(:,:,k),deriv,elem(ie)%Dinv)   

  # gradw_i(:,:,:,k)   = gradient_sphere(elem(ie)%state%w_i(:,:,k,n0),deriv,elem(ie)%Dinv)
  # v_gradw_i(:,:,k) = v_i(:,:,1,k)*gradw_i(:,:,1,k) + v_i(:,:,2,k)*gradw_i(:,:,2,k)
  # ! w - tendency on interfaces 
  # w_tens(:,:,k) = (-w_vadv_i(:,:,k) - v_gradw_i(:,:,k))*scale1 - scale2*g*(1-dpnh_dp_i(:,:,k) )

  # ! phi - tendency on interfaces
  # ! vtemp(:,:,:,k) = gradphinh_i(:,:,:,k) + &
  # !    (scale2-1)*hvcoord%hybi(k)*elem(ie)%derived%gradphis(:,:,:)
  # v_gradphinh_i(:,:,k) = v_i(:,:,1,k)*gradphinh_i(:,:,1,k) &
  #      +v_i(:,:,2,k)*gradphinh_i(:,:,2,k) 
  # phi_tens(:,:,k) =  (-phi_vadv_i(:,:,k) - v_gradphinh_i(:,:,k))*scale1 &
  #   + scale2*g*elem(ie)%state%w_i(:,:,k,n0)
  #note: use entropy stable form 
  # if (theta_advect_form==0) then
  #    v_theta(:,:,1,k)=elem(ie)%state%v(:,:,1,k,n0)*vtheta_dp(:,:,k)
  #    v_theta(:,:,2,k)=elem(ie)%state%v(:,:,2,k,n0)*vtheta_dp(:,:,k)
  #    div_v_theta(:,:,k)=divergence_sphere(v_theta(:,:,:,k),deriv,elem(ie))
  # else
  #    ! alternate form, non-conservative, better HS topography results
  #    v_theta(:,:,:,k) = gradient_sphere(vtheta(:,:,k),deriv,elem(ie)%Dinv)
  #    div_v_theta(:,:,k)=vtheta(:,:,k)*divdp(:,:,k) + &
  #         dp3d(:,:,k)*elem(ie)%state%v(:,:,1,k,n0)*v_theta(:,:,1,k) + &
  #         dp3d(:,:,k)*elem(ie)%state%v(:,:,2,k,n0)*v_theta(:,:,2,k) 
  # endif
  #   temp(:,:,k) = (elem(ie)%state%w_i(:,:,k,n0)**2 + &
  #      elem(ie)%state%w_i(:,:,k+1,n0)**2)/4
  # wvor(:,:,:,k) = gradient_sphere(temp(:,:,k),deriv,elem(ie)%Dinv)
  # wvor(:,:,1,k) = wvor(:,:,1,k) - (elem(ie)%state%w_i(:,:,k,n0)*gradw_i(:,:,1,k) +&
  #      elem(ie)%state%w_i(:,:,k+1,n0)*gradw_i(:,:,1,k+1))/2
  # wvor(:,:,2,k) = wvor(:,:,2,k) - (elem(ie)%state%w_i(:,:,k,n0)*gradw_i(:,:,2,k) +&
  #      elem(ie)%state%w_i(:,:,k+1,n0)*gradw_i(:,:,2,k+1))/2
  

  grad_kinetic_energy = sphere_gradient_nlev((u[:,:,:,:,0]**2 + u[:,:,:,:,1]**2)/2.0, h_grid, a=radius_earth)
  # KE(:,:,k) = ( elem(ie)%state%v(:,:,1,k,n0)**2 + elem(ie)%state%v(:,:,2,k,n0)**2)/2
  v_vtheta = u * vtheta_dpi[:,:,:,:,np.newaxis]
  div_v_vtheta = sphere_divergence_nlev(v_vtheta, h_grid, a=radius_earth)/2.0

  grad_vtheta = sphere_gradient_nlev(vtheta, h_grid, a=radius_earth)

  div_v_vtheta += vtheta * divdp + (dpi * (u[:,:,:,:,0] * grad_vtheta[:,:,:,:,0] + 
                                           u[:,:,:,:,1] * grad_vtheta[:,:,:,:,1]))/2.0
#      ! hydrostatic pressure
#      pi_i(:,:,1)=hvcoord%hyai(1)*hvcoord%ps0
#      do k=1,nlev
#         pi_i(:,:,k+1)=pi_i(:,:,k) + dp3d(:,:,k)
#      enddo
# #ifdef HOMMEXX_BFB_TESTING
#      do k=1,nlev
#         pi(:,:,k) = (pi_i(:,:,k+1)+pi_i(:,:,k))/2
#      enddo
#      exner  = bfb_pow(pi/p0,kappa)
# #else
#      do k=1,nlev
#         pi(:,:,k)=pi_i(:,:,k) + dp3d(:,:,k)/2
#      enddo
#      exner  = (pi/p0)**kappa
# #endif

#      pnh = pi ! copy hydrostatic pressure into output variable
#      dpnh_dp_i = 1 
#      if (present(pnh_i_out)) then  
#        pnh_i_out=pi_i 
#      endif

  # v_theta(:,:,1,k)=elem(ie)%state%v(:,:,1,k,n0)*vtheta_dp(:,:,k)
  # v_theta(:,:,2,k)=elem(ie)%state%v(:,:,2,k,n0)*vtheta_dp(:,:,k)
  # div_v_theta(:,:,k)=divergence_sphere(v_theta(:,:,:,k),deriv,elem(ie))/2

  # v_theta(:,:,:,k) = gradient_sphere(vtheta(:,:,k),deriv,elem(ie)%Dinv)

  # div_v_theta(:,:,k)=div_v_theta(:,:,k) + (  & 
  #      vtheta(:,:,k)*divdp(:,:,k) + &
  #      dp3d(:,:,k)*elem(ie)%state%v(:,:,1,k,n0)*v_theta(:,:,1,k) + &
  #      dp3d(:,:,k)*elem(ie)%state%v(:,:,2,k,n0)*v_theta(:,:,2,k)  ) /2 
  # theta_tens(:,:,k)=(-theta_vadv(:,:,k)-div_v_theta(:,:,k))*scale1
  # gradKE(:,:,:,k) = gradient_sphere(KE(:,:,k),deriv,elem(ie)%Dinv)
  grad_exner = gradient_sphere_nlev((get_p_mid/p0)**(Rgas/cp), h_grid, a=radius_earth)

#   gradpterm(:,:,1,k) = Cp*vtheta(:,:,k)*gradexner(:,:,1,k)
#   gradpterm(:,:,2,k) = Cp*vtheta(:,:,k)*gradexner(:,:,2,k)
  gradpterm = cp * vtheta[:,:,:,:,np.newaxis] * grad_exner
#   if (theta_advect_form==2) then
#      ! split form. average of default and above
#      vtemp(:,:,:,k) = gradient_sphere(vtheta(:,:,k)*exner(:,:,k),deriv,elem(ie)%Dinv)
#      v_theta(:,:,:,k) = gradient_sphere(vtheta(:,:,k),deriv,elem(ie)%Dinv)
# #ifdef HOMMEDA
#      vtemp(:,:,1,k) = vtemp(:,:,1,k) * invrhatm(:,:,k)
#      vtemp(:,:,2,k) = vtemp(:,:,2,k) * invrhatm(:,:,k)
#      v_theta(:,:,1,k) = v_theta(:,:,1,k) * invrhatm(:,:,k)
#      v_theta(:,:,2,k) = v_theta(:,:,2,k) * invrhatm(:,:,k)
# #endif
#      gradpterm(:,:,1,k) = (gradpterm(:,:,1,k) + Cp*(vtemp(:,:,1,k)-exner(:,:,k)*v_theta(:,:,1,k)))/2
#      gradpterm(:,:,2,k) = (gradpterm(:,:,2,k) + Cp*(vtemp(:,:,2,k)-exner(:,:,k)*v_theta(:,:,2,k)))/2
#   endif

  # gradexner(:,:,:,k) = gradient_sphere(exner(:,:,k),deriv,elem(ie)%Dinv)

  # mgrad(:,:,1,k) = (dpnh_dp_i(:,:,k)*gradphinh_i(:,:,1,k)+ &
  #       dpnh_dp_i(:,:,k+1)*gradphinh_i(:,:,1,k+1))/2
  # mgrad(:,:,2,k) = (dpnh_dp_i(:,:,k)*gradphinh_i(:,:,2,k)+ &
  #       dpnh_dp_i(:,:,k+1)*gradphinh_i(:,:,2,k+1))/2

  # vtens1(i,j,k) = (-v_vadv(i,j,1,k) &
  #      + v2*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
  #      - gradKE(i,j,1,k) - mgrad(i,j,1,k) &
  #     -gradpterm(i,j,1,k)&
  #     -wvor(i,j,1,k) )*scale1


  # vtens2(i,j,k) = (-v_vadv(i,j,2,k) &
  #      - v1*(elem(ie)%fcor(i,j) + vort(i,j,k)) &
  #      - gradKE(i,j,2,k) - mgrad(i,j,2,k) &
  #     -gradpterm(i,j,2,k) &
  #     -wvor(i,j,2,k) )*scale1


  # elem(ie)%state%dp3d(:,:,k,np1) = &
  #      elem(ie)%spheremp(:,:) * (scale3 * elem(ie)%state%dp3d(:,:,k,nm1) - &
  #      scale1*dt2 * (divdp(:,:,k) + eta_dot_dpdn(:,:,k+1)-eta_dot_dpdn(:,:,k)))

  pass