procedure CreateSchedule(chromosome, schedule_start_date)
  Create a new schedule object
   if AddFirstCampaign(first gene in chromosome, schedule, schedule_start_date) == true:# Add first campaign ,True=Continue building schedule False
        for each remaining gene in chromosome: #For 
            if prev_gene.product != gene.product:  # If there is product changeover
                if AddNewCampaign(gene, schedule) == false▻ product changeover:
                    break
                end if
            else:  # Without Product Changeover, add more one batch to the schedule
                if ContinuePreviousCampaign(gene, schedule) == false:
                    break:
                end if
            end if
        end for
    end if
    return schedule
end procedure

procedure AddFirstCampaign(gene, schedule, schedule_start_date)
  Create a new campaign object
   campaign.product = gene.product
    campaign.start = schedule_start_date
    campaign.first_harvest = campaign.start + USP duration of campaign.product# ?1)Shouldn´t it be Start+Inoculation? 2)What is USP duration of campaign? Is it the USP[time/batch]*batch? I believe that it is USP[time/batch]*1[batch]
    if AddFirstBatch(campaign) == false:# Adds first batch and returns false if there is no more batches to add
        return false:
    else:
        AddRemainingBatches(gene, campaign)
    end if
    Add campaign to schedule.campaigns list
    return true ▻ will signal to CreateSchedule procedure to continue building the schedule
end procedure

procedure AddNewCampaign(gene, schedule)
  Create a new campaign object # I have a new product
   prev_campaign = last most recent campaign in schedule.campaigns list
    campaign.product = gene.product
    campaign.first_harvest = prev_campaign.end + changeover duration ▻ see Figure 5.6.c #Makes sense, only start after the end 
    campaign.start = campaign.first_harvest – USP time of campaign.product #This only make sense if I add a restriction of the inoculation 1)Add a restriction or consider negligible, given that the inoculation time is fast and I would usually have more than one batch
    if AddFirstBatch(campaign) == false:# False if date of DSP end > planning horizon
        return false:
    else:
        AddRemainingBatches(gene.num_batches – 1, campaign)
    end if
    Add campaign to schedule.campaigns
    return true
end procedure

procedure ContinuePreviousCampaign(gene, schedule)
    prev_campaign = last most recent campaign in schedule.campaigns list
    return AddRemainingBatches(gene.num_batches, prev_campaign)
end procedure

procedure AddFirstBatch(campaign)
  Create a new batch object
   batch.product = campaign.product
    batch.harvested_on = campaign.first_harvest #Given that campaign.first_harvest = campaign.start + USP duration of campaign.product, harvested_on= the end of the USP of the first batch, 1) I belive batch.harvested_on=batch.first_harvest
    batch.stored_on = batch.first_harvest + DSP duration of batch.product #stored_on=USP+DSP
    if batch.stored_on > planning horizon # False if date of DSP end > planning horizon (Stops addying); else True. 1)If the batch surpass the final date it will be added, however no other batch will be added and no kg of manufacturing will be added
      return false ▻ this will send a signal to CreateSchedule procedure to stop
    end if
    batch.kg = manufacturing yield of batch.product
    batch.start = campaign.start #campaign.start=schedule_start_date
    batch.approved_on = batch.stored_on + QC/QA approval time of batch.product #batch.approved_on = USP+DSP+QC/QA
    Add batch to campaign.batches list
    Add batch to schedule.inventory for the appropriate batch.product demand due date
    campaign.kg += batch.kg
    return true
end procedure

procedure AddRemainingBatches(num_batches, campaign)
    Ensure num_batches is within the minimum and maximum batch throughput bounds #Fixes here the min and max number of batches, what happens if batches are not equal to the minimum or maximum?
    Ensure num_batches is a multiple of the given number for gene.product #Fixes here the min and max number of batches
    while num_batches > 1
      Create a new batch object
       prev_batch = last most recent batch in campaign.batches list
        batch.product = campaign.product
        batch.harvested_on = previous_batch.stored_on # batch.harvested_on=USP+DSP #batch.stored_on = batch.harvested_on + DSP time of batch.product, harvested_on is the start batch and ?end of USP?
        batch.stored_on = batch.harvested_on + DSP time of batch.product #?1)This does not make sense, I am considering the end of DSP of the last batch + 1 DSP. This only makes sense if I consider that my USP stage can be used for all subsequent batches of the same product
        if batch.stored_on > planning horizon:
            return false ▻ this will send a signal to CreateSchedule procedure to stop
        end if
        batch.kg = manufacturing yield of batch.product
        batch.start = batch.harvested_on – USP duration of batch.product #batch.harvested_on=(USP+DSP)n-1; therefore batch.start=(USPn-1+DSPn-1)-USPn and it also do not make sense It is the same to say that the batch 1 starts after DSP days.
        batch.approved_on = batch.stored_on + QC/QA approval time of batch.product
        Add batch to campaign.batches list
        Add batch to schedule.inventory for the appropriate batch.product demand due date
        campaign.kg += campaign.kg + batch.kg
        num_batches = num_batches – 1
    end while
    last_batch = last most recent batch in campaign.batches list
    campaign.end = last_batch.stored_on
end procedure
