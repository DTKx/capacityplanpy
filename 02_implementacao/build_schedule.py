procedure CreateSchedule(chromosome, schedule_start_date)
    Create a new schedule object
    if AddFirstCampaign(first gene in chromosome, schedule, schedule_start_date) == true
        for each remaining gene in chromosome:
            if prev_gene.product != gene.product

if AddNewCampaign(gene, schedule) == false
▻ product changeover

break
end if
else
if ContinuePreviousCampaign(gene, schedule) == false
break
end if
end if
end for
end if 16 return schedule 17 end procedure 18 19 procedure AddFirstCampaign(gene, schedule, schedule_start_date) 20 Create a new campaign object 21 campaign.product = gene.product 22 campaign.start = schedule_start_date 23 campaign.first_harvest = campaign.start + USP duration of campaign.product 24 if AddFirstBatch(campaign) == false 25
return false
else 27
AddRemainingBatches(gene, campaign)
end if 29 Add campaign to schedule.campaigns list
30 return true ▻ will signal to CreateSchedule procedure to continue building the schedule
31 end procedure 32 33 procedure AddNewCampaign(gene, schedule) 34 Create a new campaign object 35 prev_campaign = last most recent campaign in schedule.campaigns list 36 campaign.product = gene.product 37 campaign.first_harvest = prev_campaign.end + changeover duration
▻ see Figure 5.6.c
38 campaign.start = campaign.first_harvest – USP time of campaign.product 39 if AddFirstBatch(campaign) == false 40
return false
41 else 42
AddRemainingBatches(gene.num_batches – 1, campaign)
43 end if 44 Add campaign to schedule.campaigns 45 return true 46 end procedure 47 48 procedure ContinuePreviousCampaign(gene, schedule) 49 prev_campaign = last most recent campaign in schedule.campaigns list 50 return AddRemainingBatches(gene.num_batches, prev_campaign) 51 end procedure 52 53 procedure AddFirstBatch(campaign) 54 Create a new batch object 55 batch.product = campaign.product 56 batch.harvested_on = campaign.first_harvest 57 batch.stored_on = batch.first_harvest + DSP duration of batch.product 58 if batch.stored_on > planning horizon 59
return false ▻ this will send a signal to CreateSchedule procedure to stop
60 end if 61 batch.kg = manufacturing yield of batch.product 62 batch.start = campaign.start 63 batch.approved_on = batch.stored_on + QC/QA approval time of batch.product 64 Add batch to campaign.batches list 65 Add batch to schedule.inventory for the appropriate batch.product demand due date 66 campaign.kg += batch.kg 67 return true 68 end procedure 69 70 procedure AddRemainingBatches(num_batches, campaign) 71 Ensure num_batches is within the minimum and maximum batch throughput bounds 72 Ensure num_batches is a multiple of the given number for gene.product 73 while num_batches > 1 74
Create a new batch object
75
prev_batch = last most recent batch in campaign.batches list
76
batch.product = campaign.product
77
batch.harvested_on = previous_batch.stored_on
78
batch.stored_on = batch.harvested_on + DSP time of batch.product
79
if batch.stored_on > planning horizon
80
return false
▻ this will send a signal to CreateSchedule procedure to stop
81
end if
82
batch.kg = manufacturing yield of batch.product
83
batch.start = batch.harvested_on – USP duration of batch.product
84
batch.approved_on = batch.stored_on + QC/QA approval time of batch.product
85
Add batch to campaign.batches list
86
Add batch to schedule.inventory for the appropriate batch.product demand due date
87
campaign.kg += campaign.kg + batch.kg
88
num_batches = num_batches – 1
89 end while 90 last_batch = last most recent batch in campaign.batches list 91 campaign.end = last_batch.stored_on 92 end procedure