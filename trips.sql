use first_project
select * from duration
select * from loc
select* from payment 
select * from trips
select * from trips_details1
select * from trips_details2
select * from trips_details3
select* from trips_details4
/*total trips */
select count(distinct tripid)  from trips_details
/* validating primary key tripid*/
select tripid,count(tripid) as cnt from trips_details4
group by tripid 
having count(tripid)>1
/*total driver*/
select count(distinct driverid)  as total_drivers from trips
/*total earnings*/
select sum(fare) as fare from trips
/*total completed trips*/
select count(distinct tripid) as trips from trips
/*total searches*/
select sum(searches) as searches from trips_details4
/*total searches which got estimate */
select sum(searches) as searches from trips_details4
/*total searches which for quotes*/
select sum(searches_for_quotes) searches from trips_details4
/*total searches which got quotes*/
select  sum(searches_got_quotes) as searches from trips_details4
/*total driver cancelled*/
select count(*) - sum(driver_not_cancelled) searches from trips_details4
/*total otp entered*/
select sum(otp_entered) searches from trips_details4
/*total end ride */
select sum(end_ride) as searches from trips_details4
/*average distance per trip */
select avg(distance) from trips
/*average fare per trip*/
select avg(fare) from trips
/*distance travelled */
select sum(distance) from trips
/*which is the most used payment method */
select a.method , b.cnt from payment as a inner join 
(select faremethod,count(distinct tripid) as cnt from trips 
group by faremethod
order by count(distinct tripid ) desc limit 1 ) as b 
on a.id =b.faremethod
/*the highest payment was made through which instrument */
select a.method from payment as a inner join (select * from trips order by fare desc limit 1 ) as b
on a.id = b.faremethod

select a.method from payment as a inner join ( select faremethod, sum(fare) from trips 
group by faremethod 
order by sum(fare) desc limit 1 ) as b 
on a.id= b.faremethod
/*which two locations had the most trips*/
select loc_from,loc_to,count(distinct tripid) as cnt  from trips 
group by loc_from,loc_to 
order by cnt desc limit 2
/*top 5 earning drivers */
select driverid,sum(fare) as fare from trips 
group by driverid
order by fare 
desc limit 5
/*which duration had most trips*/
select duration,count(distinct tripid) as cnt from trips 
group by duration 
order by cnt 
desc limit 1
/*which driver ,customer pair had more orders */
select driverid,custid,count(distinct tripid) as cnt from trips 
group by driverid,custid
order by cnt 
desc limit 2
/*search to estimate rate */
select sum(searches_got_estimate)*100/sum(searches) from trips_details4
/*estimate to search for quote rates */
select sum(searches_for_quotes)*100/sum(searches_got_estimate) from trips_details4
/*quote acceptance rate*/
select sum(searches_got_quotes)*100/sum(searches_for_quotes) from trips_details4
/*booking rate */
select  sum(customer_not_cancelled)*100/sum(searches_got_quotes) as booking_good from trips_details4
/*which area got highest trips in which duration */
select duration , loc_from,count(distinct tripid) as cnt from trips 
group by duration , loc_from
order by cnt desc 

/* which area got the highest fares, cancellation ,trips */
select loc_from,sum(fare) from trips 
group by loc_from 
order by sum(fare) desc limit 1

/*highest customer cancelled areas */
select loc_from, count(*) - sum(driver_not_cancelled) as count 
from trips_details4
group by loc_from 
order by count desc limit 1 

/*which duration got the highest trips and fare */
select duration, sum(fare) as fare 
from trips
group by duration
order by  fare  desc limit 1 



