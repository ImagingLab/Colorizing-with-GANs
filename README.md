# Additive vs. Subtractive Color Spaces
Traditionally, digital color images are represented using the red, green, blue color channels (RGB). This is known as an additive color space as the source of the colors originate from emitted light. On the contrary, subtractive color spaces draw their colors from reflected light (i.e. pigments). For this project, the image data will be based on L*a*b* subtractive color space. L*a*b* contain dedicated channels depict the brightness of the image in its respective color space. The color information are fully encoded in the remaining two channels. As a result, this prevents any sudden variations in both color and brightness through small perturbations in intensity values that are experienced through RGB. 

The L*a*b* color space can be viewed as a three dimensional space with L*, a*, b* as the coordinate axis. The L* axis controls the black and white levels (L* = 0 and L* = 100 respectively), the a* axis is a gradient between green and red, and the b* axis is a gradient between blue and yellow. A visualization of the distribution of color can be seen in the following figure:
<p align='center'>  
  <img src='img/LAB.png' />
</p>
