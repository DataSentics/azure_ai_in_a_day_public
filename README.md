# azure-ai-in-a-day
Materials for seminar Azure AI in a day

## Hands-on lab environments
* Link: [https://bit.ly/3eB48mw](https://bit.ly/3eB48mw)
* Activation code: will be given on workshop day

### Software to install on the virtual machine / your local machine
* Note: you may already have this software in the lab virtual machine.
* Required: Azure Storage Explorer: [https://azure.microsoft.com/en-us/features/storage-explorer/](https://azure.microsoft.com/en-us/features/storage-explorer/)
* Optional: Postman: [https://www.postman.com/downloads/](https://www.postman.com/downloads/)

## Cognitive Services
Homepage for all CS APIs: [https://docs.microsoft.com/en-us/azure/cognitive-services](https://docs.microsoft.com/en-us/azure/cognitive-services/)[/](https://docs.microsoft.com/en-us/azure/cognitive-services/)

API for text analytics with cognitive services: [https://eastus2.dev.cognitive.microsoft.com/docs/services/TextAnalytics-v2-1/operations/56f30ceeeda5650db055a3c9/console](https://eastus2.dev.cognitive.microsoft.com/docs/services/TextAnalytics-v2-1/operations/56f30ceeeda5650db055a3c9/console)

API for image analytics: [https://eastus2.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-ga/operations/56f91f2e778daf14a499f21b/](https://eastus2.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-ga/operations/56f91f2e778daf14a499f21b/console)[console](https://eastus2.dev.cognitive.microsoft.com/docs/services/computer-vision-v3-ga/operations/56f91f2e778daf14a499f21b/console)

Car plate: [https://i1.wp.com/www.sixteen-nine.net/wp-content/uploads/2017/01/rplate.jpg](https://i1.wp.com/www.sixteen-nine.net/wp-content/uploads/2017/01/rplate.jpg)

Michael Jackson: [https://pyxis.nymag.com/v1/imgs/fd7/b0e/d23c368de349d978484d12ae1b78efd46e-21-michael-jackson-new.rsquare.w330.jpg](https://pyxis.nymag.com/v1/imgs/fd7/b0e/d23c368de349d978484d12ae1b78efd46e-21-michael-jackson-new.rsquare.w330.jpg)

### Knowledge mining with Cognitive Search
Lab instructions: [https://github.com/cynotebo/KM-Ready-Lab/tree/master/KM-Ready-Lab/workshops](https://github.com/cynotebo/KM-Ready-Lab/tree/master/KM-Ready-Lab/workshops)

Display name: clinical-trials-small

Data URI: https://kmreadylab.blob.core.windows.net/?sv=2020-02-10&ss=bfqt&srt=sco&sp=rl&se=2021-05-11T17:00:42Z&st=2021-05-10T21:00:42Z&spr=https&sig=CLTQYP92saYPSCjdFmp7B3cCWwsQ8JSNVuecI84sANk%3D

Example queries:
* `gaucher&highlight=content&$count=true`
* `gaucher&select=metadata_title,locations&facet=locations`
* `morquio&$select=metadata_title,locations&$filter=locations/any(location: location eq 'Lamy')`
* `heart&highlight=content&$filter=organizations/any(organization: organization eq 'University of California at San Diego') or locations/any(loc: loc eq 'Wisconsin')&$count=true`

## MLOps demo / lab
#### AutoML
 * go to https://github.com/solliancenet/tech-immersion-data-ai/blob/master/ai-exp3/README.md
 * follow instructions up until model training (**excluding "register model"**)
 * deploy model from Azure Machine Learning workspace (instructions will be supplied)

## Feedback form
If you can find 5-10 minutes, give us feedback, please!

[https://bit.ly/](https://forms.office.com/Pages/ResponsePage.aspx?id=qidRWJGGJU-Xd4y3jM8NkQBdrI04kUpGilB6NtMzzlBURDdaR0JXTkhZRE85MDNVWlEzWU4yVzIzUy4u)[2Z9zKaY](https://forms.office.com/Pages/ResponsePage.aspx?id=qidRWJGGJU-Xd4y3jM8NkQBdrI04kUpGilB6NtMzzlBURDdaR0JXTkhZRE85MDNVWlEzWU4yVzIzUy4u)
