//use env_logger::fmt::Color;
use log::{info, set_boxed_logger, ParseLevelError};
use std::ops::{Neg,AddAssign,MulAssign,DivAssign,Index,IndexMut,Add,Sub,Div,Mul};
use std::fmt;
use std::rc::Rc;
use rand::Rng;


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
fn random_unit_vector()->Vec3{
    loop{
        let p = Vec3::random_ranged(-1.0,1.0);
        let lensq = p.length_squared();
        if 1e-160 < lensq && lensq <= 1.0{
            return p/lensq.sqrt();
        }
    }
    Vec3::blank_vector()
}

#[inline(always)]
fn random_on_hemisphere(normal:&Vec3)-> Vec3{
    let on_unit_sphere = random_unit_vector();
    if dot_product(&on_unit_sphere,&normal) > 0.0{
        return on_unit_sphere;
    }
    -on_unit_sphere
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
    front_face:bool,
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
        front_face:false,
        }
    }
}

impl Clone for HitRecord {
    fn clone(&self) -> Self {
        Self {
            p: self.p.clone(),
            normal:self.normal.clone(),
            t:self.t.clone(),
            front_face:self.front_face.clone()
        }
    }
}

trait Hittable{
    fn hit(&self, r:&Ray,ray_t:Interval,rec:&mut HitRecord)->bool;
}


struct Sphere{
    center: Point3,
    radius:f32
}

impl Sphere{
    pub fn new(center: &Point3,radius:f32)->Sphere{
        Sphere{
            center: center.clone(),
            radius: if 0.0 >  radius {0.0} else {radius}
        }
    }
}

impl Clone for Sphere {
    fn clone(&self) -> Self {
        Self {
            center:self.center.clone(),
            radius:self.radius.clone()
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


        true

    }
}

struct HittableList{
    objects: Vec<Rc<dyn Hittable>>,
}

impl HittableList {
    pub fn default()->HittableList{
        HittableList{
            objects: Vec::new(),
        }
    }

    pub fn new(object: Rc<dyn Hittable>)->HittableList{
        HittableList{
            objects: vec![object],
        }

    }


    pub fn clear(&mut self){
        self.objects.clear();
    }

    pub fn add(&mut self, object: Rc<dyn Hittable>){
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

                for sample in 0..self.samples_per_pixel {
                    let r = self.get_ray(i,j);
                    pixel_color += self.ray_color(&r,self.max_depth,&world);
                }

                
                write_color(self.pixel_samples_scale * pixel_color);
            }
            
        }
        info!("\rDone              \n");
    }



    pub fn initialize(aspect_ratio:f32, image_width:f32,samples_per_pixel:u32)->Camera{

        let max_depth = 50;

        let image_height =  (image_width/aspect_ratio) as u32;
        let image_height = if image_height<1{1} else {image_height};

        let focal_length = 1.0;

        let viewport_height = 2.0;
        let viewport_width = viewport_height * (image_width/image_height as f32);

        let camera_center = Point3::blank_vector();


        let viewport_u = Vec3::filled_vector(viewport_width, 0.0, 0.0);
        let viewport_v = Vec3::filled_vector(0.0, -viewport_height, 0.0);

        let viewport_upper_left = camera_center.clone() - Vec3::filled_vector(0.0, 0.0, focal_length) - (viewport_u.clone()/2.0) - (viewport_v.clone()/2.0);

        let pixel_delta_u=  viewport_u.clone()/image_width;
        let pixel_delta_v = viewport_v.clone()/image_height as f32;

        
        let pixel100_loc = viewport_upper_left.clone() + (0.5*(pixel_delta_u.clone()+pixel_delta_v.clone()));

        Camera{


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
        }
    }

    fn sample_square()->Vec3{
        Vec3::filled_vector(random_double()-0.5,random_double()-0.5,0.0)
    }

    fn get_ray(&self,i:u32,j:u32)->Ray{
        let offset:Vec3 = Camera::sample_square();
        let pixel_sample = self.pixel100_loc.clone()
            + ((i as f32+offset.x()) as f32 * self.pixel_delta_u.clone()) 
            + ((j as f32+offset.y()) as f32 * self.pixel_delta_v.clone());

        let ray_origin = self.center.clone();
        let ray_direction = pixel_sample - ray_origin.clone();

        Ray::filled_ray(ray_origin,ray_direction)

    }

    

    fn ray_color(&self, r:&Ray,depth:u32, world:&HittableList)->Color{
        if depth <= 0{
            return Color::blank_vector();
        }

        let mut rec:HitRecord = HitRecord::default();

        if world.hit(r, Interval{min:0.001, max:INF}, &mut rec){
            let direction = rec.normal.clone() + random_on_hemisphere(&rec.normal);
            return 0.5*(self.ray_color(&Ray::filled_ray(rec.p,direction),depth-1,&world));
        }


        // let t =  hit_sphere(&Point3::filled_vector(0.0, 0.0, -1.0), 0.5, r);
        // if t > 0.0 {
        //     let n: Vec3 = unit_vector(r.clone().at(t)-Vec3::filled_vector(0.0, 0.0, -1.0));
        //     return 0.5*Color::filled_vector(n.x()+1.0, n.y()+1.0, n.z()+1.0)
        // }

        let unit_direction = unit_vector(r.clone().direction());
        let a = 0.5*(unit_direction.y() + 1.0);
        ((1.0-a)*Color::filled_vector(1.0, 1.0, 1.0)) + (a*Color::filled_vector(0.5, 0.7, 1.0))
        }
}



fn main(){
    env_logger::init();

    let mut world: HittableList = HittableList::default();

    world.add(Rc::new(Sphere::new(&Point3::filled_vector(0.0, 0.0, -1.0), 0.5)));
    world.add(Rc::new(Sphere::new(&Point3::filled_vector(0.0, -100.5, -1.0), 100.0)));

    let cam = Camera::initialize(16.0/9.0, 400.0,100);

    cam.render(&world);
    

    //let height = 256;
    //let width = 256;

    
    
    

    
}

