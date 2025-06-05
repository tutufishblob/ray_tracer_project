//use env_logger::fmt::Color;
//use env_logger::fmt::Color;
//use env_logger::fmt::Color;
use log::{info, set_boxed_logger, ParseLevelError};
use std::ops::{Neg,AddAssign,MulAssign,DivAssign,Index,IndexMut,Add,Sub,Div,Mul};
use std::{fmt, vec};
use std::rc::Rc;
use rand::{random, Rng};
use std::cmp::{max, min};
use std::thread;
use std::sync::Arc;

//use std::env;
//use std::fs;
//use::std::str;
//use std::io;
//use std::io::*;


const PI: f32 = 3.141592653589793;
const INF: f32 = f32::INFINITY;

struct Vec3{
    e: [f32; 3],

    
}

impl  Vec3{
    pub fn blank_vector()->Vec3{
        Vec3{
            e:[0.0,0.0,0.0]
        }
    }
    pub fn filled_vector(e1:f32, e2:f32, e3:f32)->Vec3{
        Vec3 {
            e: [e1, e2, e3]
        }
    }


    pub fn random()->Vec3{
        Vec3{
            e: [random_double(),random_double(),random_double()]
        }
    }

    pub fn random_ranged(min:f32,max:f32)->Vec3{
        Vec3{
            e: [random_double_with_range(min,max),random_double_with_range(min,max),random_double_with_range(min,max)]
        }
    }

    pub const fn x(&self)->f32{
        self.e[0]
    }

    pub const fn y(&self)->f32{
        self.e[1]
    }

    pub const fn z(&self)->f32{
        self.e[2]
    }

    pub fn length(&self)->f32{
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self)->f32{
        (self.e[0]* self.e[0]) + (self.e[1]* self.e[1]) + (self.e[2]* self.e[2])
    }

    pub fn near_zero(&self)->bool{
        let s = 1e-8;
        return if self.e[0].abs() < s && self.e[1].abs() < s && self.e[2].abs() < s {true} else {false}
    }
    
}

impl Clone for Vec3 {
    fn clone(&self) -> Self {
        Self {
            e: self.e.clone() 
        }
    }
}

impl Index<usize> for Vec3{
    type Output = f32;
    fn index(&self, index: usize)->&f32{
        &self.e[index]
    }
}

impl IndexMut<usize> for Vec3{
    //type Output = f32;
    fn index_mut(&mut self, index: usize)->&mut f32{
        &mut self.e[index]
    }
}

impl Neg for Vec3{
    type Output = Self;

    fn neg(self)->Self{
        Vec3 {e: [(-self.e[0]), (-self.e[1]), (-self.e[2])]}
    }
}

impl AddAssign for Vec3{
    //type Output = Self;

    fn add_assign(&mut self, other: Self){
        *self = Self {e: [(self.e[0]+other.e[0]), (self.e[1]+other.e[1]), (self.e[2]+other.e[2])]};
    }
}

impl DivAssign<f32> for Vec3{
    //type Output = Self;

    fn div_assign(&mut self, t: f32){
        Self {e: [(self.e[0]/t), (self.e[1]/t), (self.e[2]/t)]};
    }
}

impl MulAssign<f32> for Vec3{
    //type Output = Self;

    fn mul_assign(&mut self, t: f32){
        *self = Self {e: [(self.e[0]*t), (self.e[1]*t), (self.e[2]*t)]};
    }
}

impl Add for Vec3{
    type Output = Self;

    fn add(self,other: Self)->Self{
        Vec3 {e: [(self.e[0]+other.e[0]), (self.e[1]+other.e[1]), (self.e[2]+other.e[2])]}
    }
}

// impl Add for &Vec3{
//     type Output = Self;

//     fn add(self,other: Self)->Self{
//         &Vec3 {e: [(self.e[0]+other.e[0]), (self.e[1]+other.e[1]), (-self.e[2]+other.e[2])]}
//     }
// }

impl Sub for Vec3{
    type Output = Self;

    fn sub(self,other: Self)->Self{
        Vec3 {e: [(self.e[0]-other.e[0]), (self.e[1]-other.e[1]), (self.e[2]-other.e[2])]}
    }
}

// impl Sub for &Vec3{
//     type Output = Self;

//     fn sub(self,other: Self)->Self{
//         &Vec3 {e: [(self.e[0]-other.e[0]), (self.e[1]-other.e[1]), (-self.e[2]-other.e[2])]}
//     }
// }

// impl Div for Vec3{
//     type Output = Self;

//     fn div(self,other: Self)->Self{
//         Vec3 {e: [(self.e[0]+other.e[0]), (self.e[1]+other.e[1]), (self.e[2]+other.e[2])]}
//     }
// }

impl Mul for Vec3{
    type Output = Self;

    fn mul(self,other: Self)->Self{
        Vec3 {e: [(self.e[0]*other.e[0]), (self.e[1]*other.e[1]), (self.e[2]*other.e[2])]}
    }
}

impl Mul<Vec3> for f32{
    type Output = Vec3;

    fn mul(self, other: Vec3)->Vec3{
        Vec3 {e: [(self*other.e[0]), (self*other.e[1]), (self*other.e[2])]}
    }
}

impl Mul<f32> for Vec3{
    type Output = Vec3;

    fn mul(self, t: f32)->Vec3{
        t*self
    }
}

impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, t:f32)->Vec3{
        (1.0/t)*self
    }
}


impl Div<Vec3> for f32 {
    
    type Output = Vec3;

    fn div(self, other: Vec3)->Vec3{
        (1.0/self)*other
    }
}


// impl<[f32;3]> Deref for &Vec3{
//     type Output = [f32;3];

//     fn deref(self)->[f32;3]{
//         self.e
//     }
// }
// impl Div<&Vec3> for f32 {
    
//     type Output = Vec3;

//     fn div(self, other: &Vec3)->Vec3{
//         (1.0/self)*(*other)
//     }
// }

fn dot_product(u: &Vec3, v: &Vec3)->f32{
    (u.e[0]*v.e[0]) + (u.e[1]*v.e[1]) + (u.e[2]*v.e[2])
}

fn cross_product(u: &Vec3, v: &Vec3)->Vec3{
    Vec3{e: [(u.e[1] * v.e[2] - u.e[2] * v.e[1]),
        (u.e[2] * v.e[0] - u.e[0] * v.e[2]),
        (u.e[0] * v.e[1] - u.e[1] * v.e[0])]}
}


#[inline(always)]
fn unit_vector(v: Vec3)->Vec3{
    v.clone() / v.length()
}

#[inline(always)]
fn random_in_unit_disk()->Vec3{
    loop{
        let p = Vec3::filled_vector(random_double_with_range(-1.0, 1.0), random_double_with_range(-1.0, 1.0), 0.0);
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

#[inline(always)]
fn random_unit_vector()->Vec3{
    loop{
        let p = Vec3::random_ranged(-1.0,1.0);
        let lensq = p.length_squared();
        if 1e-160 < lensq && lensq <= 1.0{
            return p/lensq.sqrt();
        }
    }
    //Vec3::blank_vector()
}

#[inline(always)]
fn random_on_hemisphere(normal:&Vec3)-> Vec3{
    let on_unit_sphere = random_unit_vector();
    if dot_product(&on_unit_sphere,&normal) > 0.0{
        return on_unit_sphere;
    }
    -on_unit_sphere
}

#[inline(always)]
fn reflect(v:&Vec3,n:&Vec3)->Vec3{
    return v.clone()-2.0*dot_product(v, n)*n.clone();
}

#[inline(always)]

fn refract(uv:&Vec3, n: &Vec3, etai_over_etat:f32)->Vec3{
    let mut cos_theta = 1.0;
    let temp = dot_product(&-uv.clone(), n);
    if temp < cos_theta {
        cos_theta = temp;
    }
    let r_out_perp: Vec3 = etai_over_etat * (uv.clone()+(cos_theta*n.clone()));
    let r_out_parallel:Vec3 = -((1.0 - r_out_perp.length_squared()).abs().sqrt())*n.clone();

    r_out_parallel+r_out_perp
    
}

type Point3 = Vec3;
type Color = Vec3;

struct Ray{
    origin: Point3,
    dir: Vec3,
}

impl Ray{
    pub fn blank_ray()->Ray{
        Ray { origin: (Point3::blank_vector()), dir: (Vec3::blank_vector()) }
    }

    pub fn filled_ray(origin: Point3, direction: Vec3)->Ray{
        Ray { origin: (origin), dir: (direction) }
    }


    pub fn origin(self)->Point3{
        self.origin
    }

    pub fn direction(self)->Vec3{
        self.dir
    }

    pub fn at(self, t:f32)->Point3{
        return self.origin + (t*self.dir);
    }
}

impl Clone for Ray {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            origin: self.origin.clone()
        }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.e[0], self.e[1],self.e[2])
    }
}

fn write_color(pixel_color: Color){
    let mut r = pixel_color.x();
    let mut g = pixel_color.y();
    let mut b = pixel_color.z();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let intensity = Interval::new(0.000,0.999);

    let rbyte = (256.0*intensity.clamp(r)) as u32;
    let gbyte = (256.0*intensity.clamp(g)) as u32;
    let bbyte = (256.0*intensity.clamp(b)) as u32;

    println!("{} {} {}", rbyte,gbyte,bbyte);
    
}


// fn ray_color(r:&Ray,world:&HittableList)->Color{
//     let mut rec:HitRecord = HitRecord::default();
//     if world.hit(r, Interval{min:0.0, max:INF}, &mut rec){
//         return 0.5*(rec.normal+Color::filled_vector(1.0, 1.0, 1.0));
//     }


//     let t =  hit_sphere(&Point3::filled_vector(0.0, 0.0, -1.0), 0.5, r);
//     if t > 0.0 {
//         let n: Vec3 = unit_vector(r.clone().at(t)-Vec3::filled_vector(0.0, 0.0, -1.0));
//         return 0.5*Color::filled_vector(n.x()+1.0, n.y()+1.0, n.z()+1.0)
//     }

//     let unit_direction = unit_vector(r.clone().direction());
//     let a = 0.5*(unit_direction.y() + 1.0);
//     ((1.0-a)*Color::filled_vector(1.0, 1.0, 1.0)) + (a*Color::filled_vector(0.5, 0.7, 1.0))
// }

fn hit_sphere(center: &Point3,radius:f32, r: &Ray)->f32{
    let oc = center.clone() - r.clone().origin();
    let a = r.clone().direction().length_squared();
    let h = dot_product(&r.clone().direction(), &oc);
    let b = -2.0 * dot_product(&r.clone().direction(), &oc);
    let c = oc.length_squared() - (radius*radius);

    let discriminant = (b*b) - (4.0*a*c);

    if discriminant < 0.0 {
        -1.0
    }
    else{
        (h-discriminant.sqrt())/a
    }
}

struct HitRecord{
    p:Point3,
    normal:Vec3,
    t:f32,
    u:f32,
    v:f32,
    front_face:bool,
    mat: Arc<dyn Material>,
}

impl HitRecord {
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3){
        self.front_face = if dot_product(&r.clone().direction(), &outward_normal) < 0.0 {true} else{false};
        self.normal = if self.front_face {outward_normal} else {-outward_normal};
    }

    pub fn default()->HitRecord{
        HitRecord{
        p:Point3::blank_vector(),
        normal:Vec3::blank_vector(),
        t:0.0,
        u:0.0,
        v: 0.0,
        front_face:false,
        mat:Arc::new(BlankMaterial{})
        }
    }
}

impl Clone for HitRecord {
    fn clone(&self) -> Self {
        Self {
            p: self.p.clone(),
            normal:self.normal.clone(),
            t:self.t.clone(),
            u: self.u.clone(),
            v: self.v.clone(),
            front_face:self.front_face.clone(),
            mat:self.mat.clone()
        }
    }
}

trait Hittable{
    fn hit(&self, r:&Ray,ray_t:Interval,rec:&mut HitRecord)->bool;
}


struct Sphere{
    center: Point3,
    radius:f32,
    mat: Arc<dyn Material>,
}

impl Sphere{
    pub fn new(center: &Point3,radius:f32,mat:Arc<dyn Material>)->Sphere{
        Sphere{
            center: center.clone(),
            radius: if 0.0 >  radius {0.0} else {radius},
            mat: mat,
        }
    }
}

impl Clone for Sphere {
    fn clone(&self) -> Self {
        Self {
            center:self.center.clone(),
            radius:self.radius.clone(),
            mat: self.mat.clone(),
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r:&Ray,ray_t:Interval,rec:&mut HitRecord)->bool {
        let oc = self.center.clone() - r.clone().origin();
        let a = r.clone().direction().length_squared();
        let h = dot_product(&r.clone().direction(), &oc);
        //let b = -2.0 * dot_product(&r.clone().direction(), &oc);
        let c = oc.length_squared() - (self.radius*self.radius);

        let discriminant = (h*h) - (a*c);

        if discriminant < 0.0 {
            return false
        }
        let sqrtd = discriminant.sqrt();

        let mut root  = (h-sqrtd)/a;
        if !ray_t.surrounds(&root) {
            root = (h+sqrtd)/a;

            if !ray_t.surrounds(&root) {
                return false
            }

        }
        
        rec.t = root;
        rec.p = r.clone().at(rec.t);
        let outward_normal = (rec.clone().p - self.center.clone())/self.radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = self.mat.clone();

        true

    }
}

struct Plane {
    x:Interval,
    y:Interval,
    z:Interval,
}

// impl Plane {
//     pub fn new(x: Interval, y:Interval, z:Interval)->Plane{
//         Plane { x: x, y: y, z: z }
//     }

//     pub fn new_points(a:Point3, b:Point3)->Plane{
//         let temp_plane = Plane { x: Interval { min: f32::min(a[0], b[0]), max: f32::max(a[0], b[0]) },
//          y: Interval { min: f32::min(a[1], b[1]), max: f32::max(a[1], b[1]) },
//           z: Interval { min: f32::min(a[2], b[2]), max: f32::max(a[2], b[2]) } }

//         temp_plane.pad_to_minimums();

//         temp_plane
//     }



//     fn pad_to_minimums(&self) {
//         let delta = 0.0001;

//         if(self.x.size() < delta) {
//             x = x.expand(delta);
//         }
//     }
// }


struct Rectangle{
    origin: Point3,
    v_side: Vec3,
    u_side: Vec3,

    w: Vec3,
    normal: Vec3,
    d:f32,

    mat: Arc<dyn Material>
}

impl Rectangle{
    pub fn new(origin: Point3, v_side: Vec3, u_side: Vec3, mat:Arc<dyn Material>)->Rectangle{
        let n = cross_product(&u_side,&v_side);
        let norm = unit_vector(n.clone());

        let d = dot_product(&norm, &origin);

        let w = n.clone() / dot_product(&n, &n);
        Rectangle{
            origin: origin,
            v_side: v_side,
            u_side: u_side,
            mat: mat,
            w:w,
            normal: norm,
            d: d,
        }
    }

    fn is_interior(a:&f32, b:&f32, rec: &mut HitRecord) ->bool {
        let unit_interval = Interval::new(0.0, 1.0);


        if !unit_interval.clone().contains(a) || !unit_interval.clone().contains(b) {
            return false;
        }

        rec.u = a.clone();
        rec.v = b.clone();
        true
    }
    
}

impl Clone for Rectangle {
fn clone(&self) -> Self {
        Self {
            origin:self.origin.clone(),
            v_side:self.v_side.clone(),
            u_side:self.u_side.clone(),
            mat: self.mat.clone(),
            w: self.w.clone(),
            normal: self.normal.clone(),
            d: self.d.clone(),
            
        }
    }
}

impl Hittable for Rectangle {
    fn hit(&self, r:&Ray,ray_t:Interval,rec:&mut HitRecord)->bool {

        let denom = dot_product(&self.normal,&r.clone().direction());

        if denom.abs() < 1e-8 {
            return false;
        }

        let t = (self.d - dot_product(&self.normal, &r.clone().origin())) / denom;

        if !ray_t.contains(&t) {
            return false;
        }

        let intersection = r.clone().at(t);

        let planar_hitpt_vector = intersection.clone()-self.origin.clone();
        let alpha = dot_product(&self.w, &cross_product(&planar_hitpt_vector,&self.v_side));
        let beta = dot_product(&self.w, &cross_product(&self.u_side,&planar_hitpt_vector));

        if !Rectangle::is_interior(&alpha,&beta,rec) {
            return false;
        }



        rec.t = t;
        rec.p = intersection.clone();
        rec.mat = self.mat.clone();
        rec.set_face_normal(r, self.normal.clone());

        true
    }
}


#[inline(always)]
pub fn init_box(a:Point3, b:Point3,mat:Arc<dyn Material>) -> HittableList{
    let mut sides = HittableList::default();

    let min = Point3::filled_vector(f32::min(a.x(), b.x()), f32::min(a.y(), b.y()), f32::min(a.z(), b.z()));
    let max = Point3::filled_vector(f32::max(a.x(), b.x()), f32::max(a.y(), b.y()), f32::max(a.z(), b.z()));

    let dx = Vec3::filled_vector(max.x() - min.x(), 0.0, 0.0);
    let dy = Vec3::filled_vector(0.0, max.y() - min.y(), 0.0);
    let dz = Vec3::filled_vector(0.0, 0.0, max.z() - min.z());

    //world.add(Arc::new(Rectangle::new(Point3::filled_vector(555.0, 0.0, 0.0), Vec3::filled_vector(0.0, 555.0, 0.0), Vec3::filled_vector(0.0, 0.0, 555.0),green)));

    sides.add(Arc::new(Rectangle::new(Point3::filled_vector(min.x(), min.y(), max.z()), dx.clone(), dy.clone(), mat.clone())));
    sides.add(Arc::new(Rectangle::new(Point3::filled_vector(max.x(), min.y(), max.z()), -dz.clone(), dy.clone(), mat.clone())));
    sides.add(Arc::new(Rectangle::new(Point3::filled_vector(max.x(), min.y(), min.z()), -dx.clone(), dy.clone(), mat.clone())));
    sides.add(Arc::new(Rectangle::new(Point3::filled_vector(min.x(), min.y(), min.z()), dz.clone(), dy, mat.clone())));
    sides.add(Arc::new(Rectangle::new(Point3::filled_vector(min.x(), max.y(), max.z()), dx.clone(), -dz.clone(), mat.clone())));
    sides.add(Arc::new(Rectangle::new(Point3::filled_vector(min.x(), min.y(), min.z()), dx, dz, mat)));

    sides
}

struct HittableList{
    objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn default()->HittableList{
        HittableList{
            objects: Vec::new(),
        }
    }

    pub fn new(object: Arc<dyn Hittable>)->HittableList{
        HittableList{
            objects: vec![object],
        }

    }


    pub fn clear(&mut self){
        self.objects.clear();
    }

    pub fn add(&mut self, object: Arc<dyn Hittable>){
        self.objects.push(object);
    }

    pub fn hit(&self, r:&Ray,ray_t:Interval,rec:&mut HitRecord)->bool{
        let mut temp_rec: HitRecord = HitRecord::default();
        let mut hit_anything:bool = false;
        let mut closest_so_far = ray_t.max;

        for object in &self.objects{
            let object = object.as_ref();

            
            if object.hit(r,Interval{min:ray_t.min, max:closest_so_far},&mut temp_rec){
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec.clone();
            }
        }

        hit_anything

    }

}

struct Interval{
    min:f32,
    max:f32,
}

impl Interval{
    pub fn default()->Interval{
        Interval{
            min:-INF,
            max:INF,
        }
    }

    pub fn new(min:f32, max:f32)->Interval{
        Interval{
            min: min,
            max: max,
        }
    }

    pub fn contains(self, x:&f32)->bool{
        if self.min <= *x && *x <= self.max {true} else {false}
    }

    pub fn surrounds(&self, x:&f32)->bool{
        if self.min < *x && *x < self.max {true} else {false}
    }

    pub fn clamp(&self, x:f32)->f32{
        if x < self.min{
            return self.min;
        }
        if x > self.max{
            return self.max;
        }

        x

    }

    const EMPTY:Interval = Interval{min:INF,max:-INF};
    const UNIVERSE: Interval = Interval{min:-INF,max:INF};
}

impl Clone for Interval{
    fn clone(&self) -> Self {
        Self { min: self.min.clone(), max: self.max.clone() }
    }
}

#[inline(always)]
fn linear_to_gamma(linear_component:f32)->f32{
    if linear_component > 0.0{
        return linear_component.sqrt();
    }
    0.0
}

#[inline(always)]
fn degrees_to_radians(degrees:f32)->f32{
    (degrees*PI)/180.0
}

#[inline(always)]
pub fn random_double()->f32{
    let mut rng = rand::thread_rng();
    rng.gen()
}

#[inline(always)]
fn random_double_with_range(min:f32,max:f32)->f32{
    let mut rng = rand::thread_rng();
    rng.gen_range(min..max)
}




struct Camera{
    max_depth:u32,
    aspect_ratio:f32,
    image_width:f32,
    samples_per_pixel:u32,

    image_height:u32,
    center:Point3,
    pixel100_loc:Point3,
    pixel_delta_u:Vec3,
    pixel_delta_v:Vec3,

    pixel_samples_scale:f32,
    vfov:f32,

    lookfrom:Point3,
    lookat:Point3,
    vup:Vec3,

    defocus_angle:f32,
    focus_dist:f32,

    defocus_disk_u:Vec3,
    defocus_disk_v:Vec3,

    background: Color,
}

impl Camera {

    // pub fn default()->Camera{
        
    // }


    pub fn render(&self, world: &HittableList){

        



        print!("P3\n{} {}\n255\n",self.image_width,self.image_height);
    
        
            
            
            

            for j in 0..self.image_height{
            info!("\rScanlines remaining: {} ",(self.image_height-j));
            for i in 0..self.image_width as u32{

                let mut pixel_color = Color::blank_vector();

                for _sample in 0..self.samples_per_pixel {
                    let r = self.get_ray(i,j);
                    pixel_color += self.ray_color(&r,self.max_depth,&world);
                }

                
                write_color(self.pixel_samples_scale * pixel_color);
            }
            
        }


       





        

        



        


        info!("\rDone              \n");
    }



    pub fn initialize(aspect_ratio:f32, image_width:f32,samples_per_pixel:u32,vfov:f32,max_depth:u32,lookfrom:Point3,lookat:Point3,vup:Vec3,defocus_angle:f32,focus_dist:f32,backgroud:Color)->Camera{
        //let vfov = 90.0;
        


        //let max_depth = 50;

        let image_height =  (image_width/aspect_ratio) as u32;
        let image_height = if image_height<1{1} else {image_height};

        let focal_length = (lookfrom.clone()-lookat.clone()).length();


        let theta =  degrees_to_radians(vfov);
        let h = (theta/2.0).tan();


        let viewport_height = 2.0*h*focus_dist;
        let viewport_width = viewport_height * (image_width/image_height as f32);

        let camera_center = lookfrom.clone();
        
        let w = unit_vector(lookfrom.clone()-lookat.clone());
        let u = unit_vector(cross_product(&vup, &w));
        let v = cross_product(&w, &u);




        let viewport_u = viewport_width*u.clone();
        let viewport_v = viewport_height*-v.clone();

        //let viewport_upper_left = camera_center.clone() - Vec3::filled_vector(0.0, 0.0, focal_length) - (viewport_u.clone()/2.0) - (viewport_v.clone()/2.0);

        let pixel_delta_u=  viewport_u.clone()/image_width;
        let pixel_delta_v = viewport_v.clone()/image_height as f32;

        let viewport_upper_left: Vec3 = camera_center.clone() - (focus_dist*w) - viewport_u/2.0 - viewport_v/2.0;

        let defocus_radius = focus_dist * degrees_to_radians(defocus_angle/2.0).tan();
        let defocus_disk_u =u*defocus_radius;
        let defocus_disk_v = v*defocus_radius;
        

        let pixel100_loc = viewport_upper_left.clone() + (0.5*(pixel_delta_u.clone()+pixel_delta_v.clone()));

        Camera{
        defocus_angle:defocus_angle,
        focus_dist:focus_dist,


        lookfrom:lookfrom,
        lookat:lookat,
        vup:vup,


        vfov:vfov,

        max_depth:max_depth,
        
        aspect_ratio: aspect_ratio,
        image_width: image_width,
        samples_per_pixel:samples_per_pixel,


        image_height:image_height,


        center: camera_center,
       

        pixel_delta_u: pixel_delta_u,
        pixel_delta_v: pixel_delta_v,

        
        pixel100_loc: pixel100_loc,

        pixel_samples_scale:1.0/samples_per_pixel as f32,


        defocus_disk_u:defocus_disk_u,
        defocus_disk_v:defocus_disk_v,
        background:backgroud,
        }
    }

    fn sample_square()->Vec3{
        Vec3::filled_vector(random_double()-0.5,random_double()-0.5,0.0)
    }

    fn defocus_disk_sample(&self) ->Point3{
        let p = random_in_unit_disk();
        return self.center.clone() + (p.e[0]*self.defocus_disk_u.clone()) + (p.e[1]*self.defocus_disk_v.clone());
    }

    fn get_ray(&self,i:u32,j:u32)->Ray{
        let offset:Vec3 = Camera::sample_square();
        let pixel_sample = self.pixel100_loc.clone()
            + ((i as f32+offset.x()) as f32 * self.pixel_delta_u.clone()) 
            + ((j as f32+offset.y()) as f32 * self.pixel_delta_v.clone());

        let ray_origin = if self.defocus_angle <= 0.0 {self.center.clone()} else{self.defocus_disk_sample()};
        let ray_direction = pixel_sample - ray_origin.clone();

        Ray::filled_ray(ray_origin,ray_direction)

    }

    

    fn ray_color(&self, r:&Ray,depth:u32, world:&HittableList)->Color{
        if depth <= 0{
            return Color::blank_vector();
        }

        let mut rec:HitRecord = HitRecord::default();

        if !world.hit(r, Interval{min:0.001, max:INF}, &mut rec){
            
            // let direction = rec.normal.clone() + random_on_hemisphere(&rec.normal);
            // return 0.5*(self.ray_color(&Ray::filled_ray(rec.p,direction),depth-1,&world));
            return self.background.clone();
        }

        let mut scattered= Ray::blank_ray();
        let mut attenuation= Color::blank_vector();
       

        let color_from_emission = rec.mat.emitted(rec.u, rec.v, &rec.p);

        if !rec.mat.scatter(&r,&rec,&mut attenuation,&mut scattered){
            return color_from_emission;
        }
        let color_from_scatter = attenuation * self.ray_color(&scattered, depth-1, world);
        // let t =  hit_sphere(&Point3::filled_vector(0.0, 0.0, -1.0), 0.5, r);
        // if t > 0.0 {
        //     let n: Vec3 = unit_vector(r.clone().at(t)-Vec3::filled_vector(0.0, 0.0, -1.0));
        //     return 0.5*Color::filled_vector(n.x()+1.0, n.y()+1.0, n.z()+1.0)
        // }

        // let unit_direction = unit_vector(r.clone().direction());
        // let a = 0.5*(unit_direction.y() + 1.0);
        // ((1.0-a)*Color::filled_vector(1.0, 1.0, 1.0)) + (a*Color::filled_vector(0.5, 0.7, 1.0))

        color_from_emission + color_from_scatter
        }
}

trait Material{
    fn scatter(&self, r_in:&Ray,rec:&HitRecord,attenuation:&mut Color,scattered:&mut Ray)->bool;

    fn emitted(&self, u: f32, v:f32, p: &Point3) -> Color;
}

// impl Material{
//     // pub fn scatter({
//     //     false
//     // }
// }

struct Light {
    value: Color
}


impl Light {
    pub fn new(value:Color)->Light{
        Light { value: value }
    }
}

impl Material for Light {
    fn emitted(&self, u: f32, v:f32, p: &Point3) -> Color {
        self.value.clone()
    }

    fn scatter(&self, r_in:&Ray,rec:&HitRecord,attenuation:&mut Color,scattered:&mut Ray)->bool {
        false
    }
}

struct Lambertian{
    albedo:Color,

}

impl Lambertian{
    pub fn new(albedo:Color)->Lambertian{
        Lambertian{
            albedo: albedo,
        }
    }

    

}

impl Material for Lambertian {
    fn scatter(&self, _r_in:&Ray,rec:&HitRecord,attenuation:&mut Color,scattered:&mut Ray)->bool {
        let mut scatter_direction = rec.normal.clone() + random_unit_vector();
        if scatter_direction.near_zero(){
            scatter_direction = rec.normal.clone();
        }
        *scattered = Ray::filled_ray(rec.p.clone(),scatter_direction);
        *attenuation = self.albedo.clone();
        true
    }

    fn emitted(&self, u: f32, v:f32, p: &Point3) -> Color {
        Point3::blank_vector()
    }
}



struct BlankMaterial{

}

impl Material for BlankMaterial {
    fn scatter(&self, _r_in:&Ray,_rec:&HitRecord,_attenuation:&mut Color,_scattered:&mut Ray)->bool {
        false
    }
    fn emitted(&self, u: f32, v:f32, p: &Point3) -> Color {
        Point3::blank_vector()
    }
}

struct Metal{
    albedo:Color,
    fuzz:f32,
}

impl Metal{
    pub fn new(albedo:Color,fuzz:f32)->Metal{
        let mut temp = fuzz;
        if fuzz >= 1.0{
            temp = 1.0;
        }
        Metal{
            albedo: albedo,
            fuzz: temp
        }
    }



}

impl Material for Metal{
    fn scatter(&self, r_in:&Ray,rec:&HitRecord,attenuation:&mut Color,scattered:&mut Ray)->bool {
        let mut reflected = reflect(&r_in.clone().direction(), &rec.normal);
        reflected = unit_vector(reflected) + (self.fuzz*random_unit_vector());
        *scattered = Ray::filled_ray(rec.p.clone(),reflected);
        *attenuation = self.albedo.clone();
        if dot_product(&scattered.clone().direction(), &rec.normal) > 0.0 {return true} else {return false}
        true
    }
    fn emitted(&self, u: f32, v:f32, p: &Point3) -> Color {
        Point3::blank_vector()
    }
}

struct Dialectric{
    refraction_index:f32,

}

impl Dialectric{
    pub fn new(refraction_index:f32)->Dialectric{
        Dialectric{
            refraction_index:refraction_index,
        }
    }

    fn reflectance(&self, cosine: f32, refraction_index:f32)->f32{
        let mut r0 = (1.0-refraction_index)/(1.0+refraction_index);
        r0 = r0*r0;
        r0 + (1.0-r0)*(1.0-cosine).powf(5.0)
    }

    
}

impl Material for Dialectric{
    fn scatter(&self, r_in:&Ray,rec:&HitRecord,attenuation:&mut Color,scattered:&mut Ray)->bool {
        *attenuation = Color::filled_vector(1.0, 1.0, 1.0);
        let ri = if rec.front_face {1.0/self.refraction_index} else{self.refraction_index};
        let unit_direction = unit_vector(r_in.clone().direction());
        
        let cos_theta = f32::min(dot_product(&-unit_direction.clone(), &rec.normal),1.0 as f32);
        let sin_theta= (1.0-cos_theta*cos_theta).sqrt();

        let cannot_refract = if ri*sin_theta > 1.0 {true} else {false};

        let mut direction = Vec3::blank_vector();

        if cannot_refract || self.reflectance(cos_theta,ri) > random_double(){
            direction = reflect(&unit_direction, &rec.normal);
        }
        else{
            direction = refract(&unit_direction, &rec.normal, ri);
        }

        *scattered = Ray::filled_ray(rec.p.clone(), direction);

        true

    }

    fn emitted(&self, u: f32, v:f32, p: &Point3) -> Color {
        Point3::blank_vector()
    }
}

fn main(){
    env_logger::init();

    let mut world: HittableList = HittableList::default();

    // let R = (PI/4.0).cos();


    // let material_left = Rc::new(Lambertian::new(Color::filled_vector(0.0, 0.0, 1.0)));
    // let material_right = Rc::new(Lambertian::new(Color::filled_vector(1.0, 0.0, 0.0)));

    // let material_ground = Rc::new(Lambertian::new(Color::filled_vector(0.8, 0.8, 0.0)));
    // let material_center = Rc::new(Lambertian::new(Color::filled_vector(0.1, 0.2, 0.5)));
    // let material_left = Rc::new(Dialectric::new(1.5));
    // let material_bubble = Rc::new(Dialectric::new(1.0/1.5));
    // let material_right = Rc::new(Metal::new(Color::filled_vector(0.8, 0.6, 0.2),1.0));

    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(0.0, 0.0, -1.2), 0.5,material_center)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(0.0, -100.5, -1.0), 100.0,material_ground)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(-1.0, 0.0, -1.0), 0.5,material_left)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(-1.0, 0.0, -1.0), 0.4,material_bubble)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(1.0, 0.0, -1.0), 0.5,material_right)));

    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(-R, 0.0, -1.0), R,material_left)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(R, 0.0, -1.0), R,material_right)));
    //-----------------------------------------------------------------------------------------------------------------------------------------------------
    // let ground_material = Rc::new(Lambertian::new(Color::filled_vector(0.5, 0.5, 0.5)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(0.0, -1000.0, 0.0),1000.0,ground_material)));

    // for a in -11..11{
    //     for b in -11..11{
    //         let choose_mat = random_double();
    //         let center = Point3::filled_vector(a as f32 + 0.9*random_double(), 0.2, b as f32+0.9*random_double());

    //         if (center.clone() - Point3::filled_vector(4.0, 0.2, 0.0)).length() > 0.9{
    //             //let mut sphere_material = Rc::new(dyn Material);
    //             if choose_mat < 0.8 {
    //                 let albedo = Color::random() * Color::random();
    //                 let sphere_material = Rc::new(Lambertian::new(albedo));
    //                 world.add(Rc::new(Sphere::new(&center,0.2,sphere_material)));

    //             }
    //             else if choose_mat < 0.95 {
    //                 let albedo = Color::random_ranged(0.5, 1.0);
    //                 let fuzz = random_double_with_range(0.0, 0.5);
    //                 let sphere_material = Rc::new(Metal::new(albedo,fuzz));
    //                 world.add(Rc::new(Sphere::new(&center,0.2,sphere_material)));
    //             }
    //             else {
    //                 let sphere_material = Rc::new(Dialectric::new(1.5));
    //                 world.add(Rc::new(Sphere::new(&center,0.2,sphere_material)));
    //             }
    //         }
    //     }
    // }

    // let material1 = Rc::new(Dialectric::new(1.5));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(0.0, 1.0, 0.0), 1.0,material1)));

    // let material2 = Rc::new(Lambertian::new(Color::filled_vector(0.4, 0.2, 0.1)));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(-4.0, 1.0, 0.0), 1.0,material2)));

    // let material3 = Rc::new(Metal::new(Point3::filled_vector(0.7, 0.6, 0.5),0.0));
    // world.add(Rc::new(Sphere::new(&Point3::filled_vector(4.0, 1.0, 0.0), 1.0,material3)));

    // let aspect_ratio = 16.0/9.0;

    // let image_width = 1200.0;

    // let sample_per_pixel = 500;

    // let max_depth = 50;

    // let vfov = 20.0;

    // let lookfrom = Point3::filled_vector(13.0, 2.0, 3.0);
    // let lookat= Point3::filled_vector(0.0, 0.0, -1.0);
    // let vup = Vec3::filled_vector(0.0, 1.0, 0.0);

    // let defocus_angle = 0.6;
    // let focus_dist = 10.0;

    // let cam = Camera::initialize(aspect_ratio, image_width,sample_per_pixel, vfov,max_depth,lookfrom,lookat,vup,defocus_angle,focus_dist);

    // //----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // let left_red = Arc::new(Lambertian::new(Color::filled_vector(1.0, 0.2, 0.2)));
    // let back_green = Arc::new(Lambertian::new(Color::filled_vector(0.2, 1.0, 0.2)));
    // let right_blue = Arc::new(Lambertian::new(Color::filled_vector(0.2, 0.2, 1.0)));
    // let upper_orange = Arc::new(Lambertian::new(Color::filled_vector(1.0, 0.5, 0.0)));
    // let lower_teal = Arc::new(Lambertian::new(Color::filled_vector(0.2, 0.8, 0.8)));   

    // world.add(Arc::new(Rectangle::new(Point3::filled_vector(-3.0, -2.0, 5.0), Vec3::filled_vector(0.0, 0.0, -4.0), Vec3::filled_vector(0.0, 4.0, 0.0), left_red)));
    // world.add(Arc::new(Rectangle::new(Point3::filled_vector(-2.0, -2.0, 0.0), Vec3::filled_vector(4.0, 0.0, 0.0), Vec3::filled_vector(0.0, 4.0, 0.0), back_green)));
    // world.add(Arc::new(Rectangle::new(Point3::filled_vector(3.0, -2.0, 1.0), Vec3::filled_vector(0.0, 0.0, 4.0), Vec3::filled_vector(0.0, 4.0, 0.0), right_blue)));
    // world.add(Arc::new(Rectangle::new(Point3::filled_vector(-2.0, 3.0, 1.0), Vec3::filled_vector(4.0, 0.0, 0.0), Vec3::filled_vector(0.0, 0.0, 4.0), upper_orange)));
    // world.add(Arc::new(Rectangle::new(Point3::filled_vector(-2.0, -3.0, 5.0), Vec3::filled_vector(4.0, 0.0, 0.0), Vec3::filled_vector(0.0, 0.0, -4.0), lower_teal)));
    
    // let aspect_ratio = 1.0;

    // let image_width = 400.0;

    // let sample_per_pixel = 100;

    // let max_depth = 50;

    // let vfov = 80.0;

    // let lookfrom = Point3::filled_vector(0.0, 0.0, 9.0);
    // let lookat= Point3::filled_vector(0.0, 0.0, 0.0);

    // let vup = Vec3::filled_vector(0.0, 1.0, 0.0);

    // let defocus_angle = 0.0;
    // let focus_dist = 10.0;

    // let cam = Camera::initialize(aspect_ratio, image_width,sample_per_pixel, vfov,max_depth,lookfrom,lookat,vup,defocus_angle,focus_dist);

    // cam.render(&world);
    
    //----------------------------------------------------------------------------- ^ basic box render ---------------------------------------------------------------

    let red = Arc::new(Lambertian::new(Color::filled_vector(0.65, 0.05, 0.05)));
    let white = Arc::new(Lambertian::new(Color::filled_vector(0.73, 0.73, 0.73)));
    let green = Arc::new(Lambertian::new(Color::filled_vector(0.12, 0.45, 0.15)));
    let light = Arc::new(Light::new(Color::filled_vector(15.0, 15.0, 15.0)));

    // world.add(Arc::new(Sphere::new(&Point3::filled_vector(0.0, -1000.0, 0.0),1000.0,left_red.clone())));
    // world.add(Arc::new(Sphere::new(&Point3::filled_vector(0.0, 2.0, 0.0),2.0,left_red)));

    world.add(Arc::new(Rectangle::new(Point3::filled_vector(555.0, 0.0, 0.0), Vec3::filled_vector(0.0, 555.0, 0.0), Vec3::filled_vector(0.0, 0.0, 555.0),green)));
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(0.0, 0.0, 0.0), Vec3::filled_vector(0.0, 555.0, 0.0), Vec3::filled_vector(0.0, 0.0, 555.0),red)));
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(343.0, 554.0, 332.0), Vec3::filled_vector(-130.0, 0.0, 0.0), Vec3::filled_vector(0.0, 0.0, -105.0),light)));
    
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(0.0, 0.0, 0.0), Vec3::filled_vector(555.0, 0.0, 0.0), Vec3::filled_vector(0.0, 0.0, 555.0),white.clone())));
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(555.0, 555.0, 555.0), Vec3::filled_vector(-555.0, 0.0, 0.0), Vec3::filled_vector(0.0, 0.0, -555.0),white.clone())));
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(0.0, 0.0, 555.0), Vec3::filled_vector(555.0, 0.0, 0.0), Vec3::filled_vector(0.0, 555.0, 0.0),white.clone())));
    
    
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(280.50, 0.0, 295.60), Vec3::filled_vector(159.45, 0.0, -44.71), Vec3::filled_vector(28.95, 0.0, 165.40), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(280.50, 330.0, 295.60), Vec3::filled_vector(159.45, 0.0, -44.71), Vec3::filled_vector(28.95, 0.0, 165.40), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(309.45, 0.0, 461.00), Vec3::filled_vector(159.45, 0.0, -44.71), Vec3::filled_vector(0.0, 330.0, 0.0), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(280.50, 0.0, 295.60), Vec3::filled_vector(159.45, 0.0, -44.71), Vec3::filled_vector(0.0, 330.0, 0.0), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(280.50, 0.0, 295.60), Vec3::filled_vector(28.95, 0.0, 165.40), Vec3::filled_vector(0.0, 330.0, 0.0), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(439.95, 0.0, 250.89), Vec3::filled_vector(28.95, 0.0, 165.40), Vec3::filled_vector(0.0, 330.0, 0.0), white.clone())));

    //
    world.add(Arc::new(Rectangle::new(Point3::filled_vector(130.00, 0.0, 65.00), Vec3::filled_vector(156.32, 0.0, 48.06), Vec3::filled_vector(-50.93, 0.0, 161.15), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(130.00, 165.0, 65.00), Vec3::filled_vector(156.32, 0.0, 48.06), Vec3::filled_vector(-50.93, 0.0, 161.15), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(79.07, 0.0, 226.15), Vec3::filled_vector(156.32, 0.0, 48.06), Vec3::filled_vector(0.0, 165.0, 0.0), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(130.00, 0.0, 65.00), Vec3::filled_vector(156.32, 0.0, 48.06), Vec3::filled_vector(0.0, 165.0, 0.0), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(130.00, 0.0, 65.00), Vec3::filled_vector(-50.93, 0.0, 161.15), Vec3::filled_vector(0.0, 165.0, 0.0), white.clone())));
world.add(Arc::new(Rectangle::new(Point3::filled_vector(286.32, 0.0, 113.06), Vec3::filled_vector(-50.93, 0.0, 161.15), Vec3::filled_vector(0.0, 165.0, 0.0), white.clone())));

    
    

    let aspect_ratio = 1.0;

    let image_width = 600.0;

    let sample_per_pixel = 10000;

    let max_depth = 50;

    let vfov = 40.0;

    let lookfrom = Point3::filled_vector(278.0, 278.0, -800.0);
    let lookat= Point3::filled_vector(278.0, 278.0, 0.0);

    let vup = Vec3::filled_vector(0.0, 1.0, 0.0);

    let defocus_angle = 0.0;
    let focus_dist = 10.0;

    let background = Color::filled_vector(0.0, 0.0, 0.0);

    let cam = Camera::initialize(aspect_ratio, image_width,sample_per_pixel, vfov,max_depth,lookfrom,lookat,vup,defocus_angle,focus_dist,background);

    cam.render(&world);

    //let height = 256;
    //let width = 256;

    
    
    

    
}

